"""LLM decision functions for the investigation engine.

All LLM-based decisions are made here. Functions are organized by model tier:
- WORKER (LITE): Quick decisions, file picking, sufficiency checks, fact extraction
- MID (FLASH): Strategic analysis, planning, replanning
- HIGH (PRO): Final synthesis only

Each function takes a GeminiClient and returns structured data.
"""

import json
import logging
import time
from typing import Optional, Any

from ..core.models import GeminiClient, ModelTier, SYSTEM_PROMPT_PRO
from ..core.search import SearchHit, SearchResults
from . import prompts

logger = logging.getLogger(__name__)


def _log_llm_call(func_name: str, tier: ModelTier, prompt_preview: str, start_time: float):
    """Log LLM call start."""
    logger.info(f"🤖 LLM_CALL: {func_name} [tier={tier.value}]")
    logger.debug(f"   Prompt preview: {prompt_preview[:150]}...")


def _log_llm_result(func_name: str, result: Any, duration: float):
    """Log LLM call result."""
    if isinstance(result, dict):
        result_preview = str(result)[:200]
    elif isinstance(result, list):
        result_preview = f"[{len(result)} items]"
    elif isinstance(result, str):
        result_preview = result[:200]
    elif isinstance(result, bool):
        result_preview = str(result)
    else:
        result_preview = str(type(result))

    logger.info(f"✅ LLM_DONE: {func_name} [{duration:.2f}s] -> {result_preview}")


# =============================================================================
# JSON PARSING HELPERS
# =============================================================================

def parse_json_safe(text: str) -> Optional[dict]:
    """Safely parse JSON from LLM response."""
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to extract JSON from generic code blocks
    if "```" in text:
        start = text.find("```") + 3
        # Skip language identifier if present
        newline = text.find("\n", start)
        if newline > start:
            start = newline + 1
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass

    return None


def extract_list_from_response(text: str) -> list[str]:
    """Extract a list of items from LLM response (one per line or comma-separated)."""
    if not text:
        return []

    # Try comma-separated first
    if "," in text and "\n" not in text.strip():
        return [item.strip() for item in text.split(",") if item.strip()]

    # Otherwise, split by newlines
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        # Remove common prefixes like "- ", "* ", "1. ", etc.
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".):":
            line = line[3:].strip()
        if line:
            items.append(line)

    return items


def _coerce_bool(value: any) -> bool:
    """Coerce a value to boolean, handling string representations.

    LLMs sometimes return "true"/"false" as strings instead of JSON booleans.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


def _coerce_int(value: any) -> int | None:
    """Coerce a value to int, handling string representations.

    LLMs sometimes return numbers as strings.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, float):
        return int(value)
    return None


# =============================================================================
# WORKER TIER DECISIONS (LITE model)
# =============================================================================

async def classify_query_complexity(
    query: str,
    client: GeminiClient,
) -> bool:
    """Classify if query needs complex (PRO) or simple (FLASH) synthesis.

    Uses LITE model to intelligently decide routing.
    Returns True if complex (use PRO), False if simple (use FLASH).
    """
    start_time = time.time()
    logger.info(f"🏷️ classify_query_complexity: analyzing '{query[:50]}...'")

    prompt = prompts.P_CLASSIFY_QUERY.format(query=query)

    _log_llm_call("classify_query_complexity", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)

    response_upper = response.strip().upper()
    is_complex = "COMPLEX" in response_upper

    result = "COMPLEX" if is_complex else "SIMPLE"
    logger.info(f"🏷️ Query classified as {result}")
    _log_llm_result("classify_query_complexity", result, time.time() - start_time)

    return is_complex


async def pick_relevant_files(
    query: str,
    files: list[str],
    client: GeminiClient,
) -> list[str]:
    """Pick most relevant files for a query. Uses LITE model."""
    start_time = time.time()
    logger.info(f"🔍 pick_relevant_files: {len(files)} files to consider")

    if not files:
        return []

    # Format file list
    file_list = "\n".join(files[:100])  # Limit to prevent token overflow

    prompt = prompts.P_PICK_FILES.format(
        query=query,
        file_list=file_list,
    )

    _log_llm_call("pick_relevant_files", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    selected = extract_list_from_response(response)

    # Filter to only files that actually exist in the list
    valid_files = [f for f in selected if f in files]

    # If nothing matched, return first few files as fallback
    if not valid_files:
        logger.warning(f"   No valid files matched, using fallback (first 5)")
        valid_files = files[:5]

    result = valid_files[:10]
    _log_llm_result("pick_relevant_files", result, time.time() - start_time)
    return result


async def prioritize_documents(
    query: str,
    candidate_files: list[str],
    already_read: list[str],
    key_issues: list[str],
    client: GeminiClient,
    max_candidates: int = 50,
) -> list[str]:
    """Dynamically rank candidate documents by relevance using LLM. Uses LITE model.

    Returns files ordered by relevance, with unread files prioritized.
    """
    start_time = time.time()
    logger.info(f"📊 prioritize_documents: {len(candidate_files)} candidates")

    if not candidate_files:
        return []

    # Deduplicate candidates while preserving order
    seen: set[str] = set()
    candidates: list[str] = []
    for f in candidate_files:
        if f not in seen:
            seen.add(f)
            candidates.append(f)

    already_read = already_read or []
    already_read_set = set(already_read)
    key_issues = key_issues or []

    prompt = prompts.P_PRIORITIZE_DOCUMENTS.format(
        query=query,
        key_issues="\n".join(f"- {issue}" for issue in key_issues) if key_issues else "None identified yet",
        candidate_files="\n".join(candidates[:max_candidates]),
        already_read="\n".join(already_read) if already_read else "None",
    )

    _log_llm_call("prioritize_documents", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    ranked_files: list[str] = []
    if result:
        ranked = result.get("ranked_files") or result.get("files") or result.get("ranking")
        if isinstance(ranked, list):
            for item in ranked:
                if isinstance(item, dict):
                    file = item.get("file") or item.get("filename") or item.get("path")
                elif isinstance(item, str):
                    file = item
                else:
                    continue
                if file and file in seen and file not in ranked_files:
                    ranked_files.append(file)

    # Fallback: try extracting from response as list
    if not ranked_files:
        ranked_files = [f for f in extract_list_from_response(response) if f in seen]

    # Final fallback: return unread candidates first, then read ones
    if not ranked_files:
        unread = [f for f in candidates[:max_candidates] if f not in already_read_set]
        read = [f for f in candidates[:max_candidates] if f in already_read_set]
        ranked_files = unread + read

    # Add any missing candidates at the end
    ranked_set = set(ranked_files)
    for f in candidates:
        if f not in ranked_set:
            if f not in already_read_set:
                ranked_files.append(f)
                ranked_set.add(f)
    for f in candidates:
        if f not in ranked_set:
            ranked_files.append(f)
            ranked_set.add(f)

    _log_llm_result("prioritize_documents", ranked_files[:10], time.time() - start_time)
    return ranked_files


async def pick_relevant_hits(
    query: str,
    results: SearchResults,
    client: GeminiClient,
    max_to_show: int = 30,
) -> list[SearchHit]:
    """Pick most relevant search hits. Uses LITE model."""
    start_time = time.time()
    logger.info(f"🔍 pick_relevant_hits: {len(results.hits)} hits to consider")

    if not results.hits:
        return []

    # Format hits for LLM
    hits_text = results.format_for_llm(max_hits=max_to_show)

    prompt = prompts.P_PICK_HITS.format(
        query=query,
        hits=hits_text,
    )

    _log_llm_call("pick_relevant_hits", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    logger.debug(f"   LLM response: {response[:200]}")

    # Parse numbers from response
    selected_indices = []
    for item in extract_list_from_response(response):
        # Extract numbers from items like "[1]", "1", "1.", etc.
        num_str = ''.join(c for c in item if c.isdigit())
        if num_str:
            try:
                idx = int(num_str) - 1  # Convert to 0-indexed
                if 0 <= idx < len(results.hits):
                    selected_indices.append(idx)
            except ValueError:
                pass

    # Return selected hits
    if selected_indices:
        result = [results.hits[i] for i in selected_indices]
        _log_llm_result("pick_relevant_hits", f"Selected {len(result)} hits", time.time() - start_time)
        return result

    # Fallback: return first few hits
    logger.warning("   No indices parsed, using fallback (first 10)")
    return results.hits[:10]


async def is_sufficient(
    query: str,
    findings: str,
    client: GeminiClient,
) -> bool:
    """Check if gathered evidence is sufficient. Uses LITE model."""
    start_time = time.time()
    logger.info(f"🤔 is_sufficient: checking if we have enough evidence")

    prompt = prompts.P_IS_SUFFICIENT.format(
        query=query,
        findings=findings,
    )

    _log_llm_call("is_sufficient", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = "yes" in response.lower()

    _log_llm_result("is_sufficient", f"{'SUFFICIENT' if result else 'NEED MORE'}", time.time() - start_time)
    logger.info(f"   LLM said: {response[:100]}")
    return result


async def should_replan(
    query: str,
    plan: str,
    results: str,
    client: GeminiClient,
) -> bool:
    """Check if we should replan the investigation. Uses LITE model."""
    prompt = prompts.P_SHOULD_REPLAN.format(
        query=query,
        plan=plan,
        results=results,
    )

    response = await client.complete(prompt, tier=ModelTier.LITE)
    return "replan" in response.lower()


# Useless search terms to filter out
USELESS_TERMS = {
    # Instruction verbs
    "analyze", "explain", "describe", "summarize", "find", "identify", "list",
    "discuss", "compare", "evaluate", "assess", "review", "examine", "determine",
    "outline", "define", "clarify", "elaborate", "investigate", "explore",
    # Question words
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    # Common words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    # Generic terms
    "information", "document", "documents", "data", "work", "thing", "things",
    "file", "files", "content", "details", "overview", "summary", "analysis",
    "report", "reports", "item", "items", "point", "points", "aspect", "aspects",
}


def filter_search_terms(terms: list[str]) -> list[str]:
    """Filter out useless search terms."""
    filtered = []
    for term in terms:
        term_lower = term.lower().strip()
        # Skip if it's a useless term
        if term_lower in USELESS_TERMS:
            continue
        # Skip if too short (less than 2 chars)
        if len(term_lower) < 2:
            continue
        # Skip if it's just a number
        if term_lower.isdigit():
            continue
        # Skip template-style queries with brackets (e.g., "[specific claim]")
        if '[' in term or ']' in term:
            logger.warning(f"Filtering out template-style query: {term}")
            continue
        filtered.append(term)
    return filtered


def filter_external_queries(queries: list[str]) -> list[str]:
    """Filter out template-style external search queries."""
    filtered = []
    for query in queries:
        # Skip template-style queries with brackets
        if '[' in query or ']' in query:
            logger.warning(f"Filtering out template-style external query: {query}")
            continue
        # Skip empty or too short queries
        if not query or len(query.strip()) < 5:
            continue
        filtered.append(query)
    return filtered


async def extract_search_terms(
    query: str,
    client: GeminiClient,
) -> list[str]:
    """Extract search terms from a query. Uses LITE model."""
    prompt = prompts.P_EXTRACT_SEARCH_TERMS.format(query=query)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    terms = extract_list_from_response(response)
    terms = filter_search_terms(terms)  # Filter out useless terms
    return terms[:10]  # Limit to 10 terms


async def classify_query(
    query: str,
    client: GeminiClient,
) -> str:
    """Classify query complexity. Uses LITE model. Returns: SIMPLE, ANALYTICAL, or COMPLEX."""
    prompt = prompts.P_CLASSIFY_QUERY.format(query=query)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    response_upper = response.upper().strip()

    if "SIMPLE" in response_upper:
        return "SIMPLE"
    elif "COMPLEX" in response_upper:
        return "COMPLEX"
    else:
        return "ANALYTICAL"


async def decide_next_action(
    query: str,
    state_summary: dict,
    client: GeminiClient,
) -> dict:
    """Decide the next action in the investigation. Uses LITE model."""
    prompt = prompts.P_DECIDE_ACTION.format(
        query=query,
        files_searched=state_summary.get("files_searched", []),
        docs_read=state_summary.get("docs_read", []),
        num_facts=state_summary.get("num_facts", 0),
        findings_summary=state_summary.get("findings_summary", "None yet"),
    )

    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result and "action" in result:
        return result

    # Fallback: if we have facts, finish; otherwise search
    if state_summary.get("num_facts", 0) > 0:
        return {"action": "done", "params": {}, "reason": "Have enough information"}
    return {"action": "search", "params": {"query": query}, "reason": "Need to search"}


# =============================================================================
# MID TIER DECISIONS (FLASH model)
# =============================================================================

async def assess_small_repo(
    query: str,
    content: str,
    client: GeminiClient,
) -> dict:
    """Unified assessment for small repositories.

    Determines:
    1. Query complexity (simple vs complex) for synthesis tier selection
    2. Whether external search is needed (with specific searches if so)

    Uses FLASH model for better judgment on external search decisions.
    The FLASH system prompt already includes guidance on being selective
    about external research.
    """
    start_time = time.time()
    content_preview = f"{len(content):,} chars"
    logger.info(f"📊 assess_small_repo: evaluating query against {content_preview}")

    prompt = prompts.P_ASSESS_SMALL_REPO.format(
        query=query,
        content=content,
    )

    _log_llm_call("assess_small_repo", ModelTier.FLASH, prompt, start_time)
    # No timeout - let the model take as long as needed for full document assessment
    response = await client.complete(prompt, tier=ModelTier.FLASH, timeout=0)
    result = parse_json_safe(response)

    if result:
        # Filter out template-style external queries if any were generated
        if "case_law_searches" in result:
            result["case_law_searches"] = filter_external_queries(result.get("case_law_searches", []))
        if "web_searches" in result:
            result["web_searches"] = filter_external_queries(result.get("web_searches", []))

        # Ensure consistency: if can_answer_from_docs is True, searches must be empty
        if result.get("can_answer_from_docs", True):
            result["case_law_searches"] = []
            result["web_searches"] = []

        complexity = result.get("complexity", "complex")
        can_answer = result.get("can_answer_from_docs", True)
        searches = len(result.get("case_law_searches", [])) + len(result.get("web_searches", []))
        logger.info(f"   Assessment: complexity={complexity}, can_answer_from_docs={can_answer}, searches={searches}")
        _log_llm_result("assess_small_repo", result, time.time() - start_time)
        return result

    # Fallback: assume complex, can answer from docs (conservative - no external search)
    logger.warning("   JSON parsing failed, using conservative fallback (no external search)")
    return {
        "complexity": "complex",
        "can_answer_from_docs": True,
        "reasoning": "Fallback assessment",
        "gap": "",
        "case_law_searches": [],
        "web_searches": [],
    }


async def check_search_sufficiency(
    query: str,
    original_gap: str,
    results_summary: str,
    client: GeminiClient,
) -> dict:
    """Check if external search results are sufficient or if more search is needed.

    Uses FLASH model to evaluate whether the search results fill the identified gap.
    Only recommends additional search if there's a critical missing piece.
    """
    start_time = time.time()
    logger.info(f"🔍 check_search_sufficiency: evaluating search results against gap")

    prompt = prompts.P_CHECK_SEARCH_SUFFICIENCY.format(
        query=query,
        original_gap=original_gap,
        results_summary=results_summary,
    )

    _log_llm_call("check_search_sufficiency", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        # Filter additional search if present
        if "additional_search" in result and result["additional_search"]:
            filtered = filter_external_queries([result["additional_search"]])
            result["additional_search"] = filtered[0] if filtered else ""

        sufficient = result.get("sufficient", True)
        logger.info(f"   Sufficiency: {sufficient}")
        _log_llm_result("check_search_sufficiency", result, time.time() - start_time)
        return result

    # Fallback: assume sufficient (conservative - no more searching)
    logger.warning("   JSON parsing failed, assuming sufficient")
    return {
        "sufficient": True,
        "reasoning": "Fallback - proceeding with available results",
        "if_not_sufficient_what_missing": "",
        "additional_search": "",
    }


async def create_plan(
    query: str,
    file_list: str,
    total_files: int,
    client: GeminiClient,
) -> dict:
    """Create an investigation plan. Uses FLASH model."""
    start_time = time.time()
    logger.info(f"📋 create_plan: creating investigation strategy for {total_files} files")

    prompt = prompts.P_CREATE_PLAN.format(
        query=query,
        file_list=file_list,
        total_files=total_files,
    )

    _log_llm_call("create_plan", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        # Filter out useless search terms
        if "search_terms" in result:
            result["search_terms"] = filter_search_terms(result["search_terms"])
        # Filter out template-style external queries
        if "case_law_searches" in result:
            result["case_law_searches"] = filter_external_queries(result["case_law_searches"])
        if "web_searches" in result:
            result["web_searches"] = filter_external_queries(result["web_searches"])
        logger.info(f"   Plan: {len(result.get('search_terms', []))} search terms, {len(result.get('priority_files', []))} priority files")
        _log_llm_result("create_plan", result, time.time() - start_time)
        return result

    # Fallback plan - also filter
    logger.warning("   JSON parsing failed, using fallback plan")
    fallback_terms = filter_search_terms(query.split()[:5])
    return {
        "key_issues": [query],
        "priority_files": [],
        "search_terms": fallback_terms if fallback_terms else ["document"],
        "success_criteria": "Find information relevant to the query",
    }


async def analyze_results(
    query: str,
    results: SearchResults,
    client: GeminiClient,
) -> dict:
    """Analyze search results and extract information. Uses FLASH model."""
    start_time = time.time()
    logger.info(f"🔬 analyze_results: analyzing {len(results.hits)} search hits")

    prompt = prompts.P_ANALYZE_RESULTS.format(
        query=query,
        results=results.format_for_llm(max_hits=30),
    )

    _log_llm_call("analyze_results", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        logger.info(f"   Found: {len(result.get('facts', []))} facts, {len(result.get('read_deeper', []))} docs to read deeper")
        _log_llm_result("analyze_results", result, time.time() - start_time)
        return result

    logger.warning("   JSON parsing failed, returning empty analysis")
    return {
        "facts": [],
        "citations": [],
        "read_deeper": [],
        "additional_searches": [],
    }


async def extract_facts(
    query: str,
    filename: str,
    content: str,
    client: GeminiClient,
    max_content_chars: int = 35000,  # Increased for full legal doc coverage
) -> dict:
    """Extract facts from a document. Uses LITE model for cost efficiency."""
    start_time = time.time()
    logger.info(f"📄 extract_facts: extracting from '{filename}' ({len(content)} chars)")

    # Truncate content if too long
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n... [truncated]"
        logger.debug(f"   Content truncated to {max_content_chars} chars")

    prompt = prompts.P_EXTRACT_FACTS.format(
        query=query,
        filename=filename,
        content=content,
    )

    _log_llm_call("extract_facts", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result:
        logger.info(f"   Extracted: {len(result.get('facts', []))} facts, {len(result.get('quotes', []))} quotes")
        _log_llm_result("extract_facts", result, time.time() - start_time)
        return result

    logger.warning("   JSON parsing failed, returning empty extraction")
    return {
        "facts": [],
        "quotes": [],
        "references": [],
    }


async def replan(
    query: str,
    previous_approach: str,
    findings: str,
    client: GeminiClient,
) -> dict:
    """Create a new investigation approach. Uses FLASH model."""
    prompt = prompts.P_REPLAN.format(
        query=query,
        previous_approach=previous_approach,
        findings=findings,
    )

    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        return result

    return {
        "diagnosis": "Previous approach did not find enough information",
        "new_approach": "Try broader search terms",
        "search_terms": query.split()[:3],
        "files_to_check": [],
    }


# =============================================================================
# HIGH TIER DECISIONS (PRO model)
# =============================================================================

async def synthesize(
    query: str,
    evidence: str,
    client: GeminiClient,
    external_research: str = "",
    pinned_content: str = "",
    tier: ModelTier = ModelTier.PRO,
) -> str:
    """Synthesize final answer.

    Uses PRO model by default. For simple queries, can use FLASH model
    but ALWAYS uses PRO system prompt for synthesis quality.
    """
    start_time = time.time()
    evidence_lines = evidence.count('\n') + 1 if evidence else 0
    pinned_info = f", {len(pinned_content)} chars pinned" if pinned_content else ""
    tier_label = "FLASH" if tier == ModelTier.FLASH else "PRO"
    logger.info(f"📝 synthesize [{tier_label}]: creating final answer from {evidence_lines} evidence lines{pinned_info}")

    prompt = prompts.P_SYNTHESIZE.format(
        query=query,
        evidence=evidence or "No specific findings accumulated.",
        external_research=external_research or "No external research conducted.",
        pinned_content=pinned_content or "No decisive documents identified.",
    )

    _log_llm_call("synthesize", tier, prompt, start_time)
    # ALWAYS use PRO system prompt for synthesis, regardless of model tier
    response = await client.complete(prompt, tier=tier, system_prompt=SYSTEM_PROMPT_PRO)

    logger.info(f"✨ Synthesis complete: {len(response)} chars")
    _log_llm_result("synthesize", f"{len(response)} char response", time.time() - start_time)
    return response


async def resolve_contradictions(
    query: str,
    contradictions: str,
    client: GeminiClient,
) -> dict:
    """Analyze contradictory evidence. Uses PRO model."""
    prompt = prompts.P_RESOLVE_CONTRADICTIONS.format(
        query=query,
        contradictions=contradictions,
    )

    response = await client.complete(prompt, tier=ModelTier.PRO)
    result = parse_json_safe(response)

    if result:
        return result

    return {
        "is_true_contradiction": False,
        "analysis": "Unable to analyze contradictions",
        "resolution": "Present both perspectives",
        "preferred_interpretation": "None",
    }


# =============================================================================
# EXTERNAL SEARCH DECISIONS
# =============================================================================

async def should_search_external(
    query: str,
    facts_found: list[str],
    key_issues: list[str],
    client: GeminiClient,
) -> dict:
    """Decide if we should search external sources (case law, web). Uses LITE model."""
    start_time = time.time()
    logger.info(f"🌐 should_search_external: evaluating need for external research")

    prompt = prompts.P_SHOULD_SEARCH_EXTERNAL.format(
        query=query,
        facts_found="\n".join(f"- {fact}" for fact in facts_found[:15]) if facts_found else "None yet",
        key_issues="\n".join(f"- {issue}" for issue in key_issues) if key_issues else "None identified",
    )

    _log_llm_call("should_search_external", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result:
        _log_llm_result("should_search_external", result, time.time() - start_time)
        return result

    # Default: search both if query seems legal in nature
    legal_keywords = ["contract", "damages", "liability", "breach", "negligence", "warranty", "claim"]
    is_legal = any(kw in query.lower() for kw in legal_keywords)

    return {
        "search_case_law": is_legal,
        "case_law_queries": [query] if is_legal else [],
        "search_web": is_legal,
        "web_queries": [query] if is_legal else [],
        "reasoning": "Default behavior based on legal keyword detection",
    }


async def analyze_case_law_results(
    query: str,
    case_law_results: str,
    client: GeminiClient,
) -> dict:
    """Analyze case law search results. Uses FLASH model."""
    start_time = time.time()
    logger.info(f"⚖️ analyze_case_law_results: extracting legal precedents")

    prompt = prompts.P_ANALYZE_CASE_LAW.format(
        query=query,
        case_law_results=case_law_results,
    )

    _log_llm_call("analyze_case_law_results", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        _log_llm_result("analyze_case_law_results", result, time.time() - start_time)
        return result

    return {
        "key_precedents": [],
        "legal_standards": [],
        "summary": "Unable to analyze case law results",
    }


async def analyze_web_results(
    query: str,
    web_results: str,
    client: GeminiClient,
) -> dict:
    """Analyze web search results for regulations/standards. Uses FLASH model."""
    start_time = time.time()
    logger.info(f"🌍 analyze_web_results: extracting regulatory information")

    prompt = prompts.P_ANALYZE_WEB_RESULTS.format(
        query=query,
        web_results=web_results,
    )

    _log_llm_call("analyze_web_results", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        _log_llm_result("analyze_web_results", result, time.time() - start_time)
        return result

    return {
        "regulations": [],
        "standards": [],
        "summary": "Unable to analyze web results",
    }


async def generate_external_queries(
    query: str,
    facts: list[str],
    entities: list[str],
    client: GeminiClient,
    triggers: str = "",
) -> dict:
    """Generate external search queries from accumulated facts and triggers. Uses LITE model.

    This is a CONSOLIDATED function that replaces the two-step approach of:
    1. "Should we search external?"
    2. "What should we search for?"

    Instead, in one call, LITE analyzes the facts and triggers and either:
    - Returns specific search queries (meaning external search IS needed)
    - Returns empty lists (meaning no external search needed)

    Args:
        query: The user's original query
        facts: Facts accumulated from document analysis
        entities: Entities identified (party names, legal terms, etc.)
        client: GeminiClient instance
        triggers: Formatted string of research triggers from document analysis

    Returns:
        Dict with case_law_queries, web_queries, and reasoning
    """
    start_time = time.time()
    trigger_count = len(triggers.split('\n')) if triggers and triggers != "None identified" else 0
    logger.info(f"🔍 generate_external_queries: analyzing {len(facts)} facts + {trigger_count} trigger categories")

    # Format facts and entities
    facts_text = "\n".join(f"- {fact}" for fact in facts[:15]) if facts else "No facts gathered yet"
    entities_text = "\n".join(f"- {entity}" for entity in entities[:10]) if entities else "None identified"
    triggers_text = triggers if triggers else "None identified"

    prompt = prompts.P_GENERATE_EXTERNAL_QUERIES.format(
        query=query,
        facts=facts_text,
        entities=entities_text,
        triggers=triggers_text,
    )

    _log_llm_call("generate_external_queries", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result:
        # Filter out any template-style queries that slipped through
        if "case_law_queries" in result:
            result["case_law_queries"] = filter_external_queries(result.get("case_law_queries", []))
        if "web_queries" in result:
            result["web_queries"] = filter_external_queries(result.get("web_queries", []))

        total_queries = len(result.get("case_law_queries", [])) + len(result.get("web_queries", []))
        _log_llm_result("generate_external_queries", f"{total_queries} queries generated", time.time() - start_time)
        return result

    logger.warning("   JSON parsing failed, returning no external queries")
    return {
        "case_law_queries": [],
        "web_queries": [],
        "reasoning": "Unable to parse response",
    }


async def extract_triggers_from_content(
    content: str,
    client: GeminiClient,
    max_chars: int = 15000,
) -> dict:
    """Extract research triggers from bulk content. Uses LITE model.

    Used in small repo mode where all content is loaded at once.
    Scans content for jurisdictions, regulations, legal doctrines, etc.

    Args:
        content: Document content to scan
        client: GeminiClient instance
        max_chars: Maximum content chars to process

    Returns:
        Dict with trigger categories and their values
    """
    start_time = time.time()
    logger.info(f"🔍 extract_triggers_from_content: scanning {len(content)} chars for triggers")

    # Truncate content if needed
    content_excerpt = content[:max_chars] if len(content) > max_chars else content

    prompt = prompts.P_EXTRACT_TRIGGERS.format(content=content_excerpt)

    _log_llm_call("extract_triggers_from_content", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result:
        # Count total triggers found
        total = sum(len(v) for v in result.values() if isinstance(v, list))
        _log_llm_result("extract_triggers_from_content", f"{total} triggers found", time.time() - start_time)
        return result

    logger.warning("   JSON parsing failed, returning empty triggers")
    return {
        "jurisdictions": [],
        "regulations_statutes": [],
        "legal_doctrines": [],
        "industry_standards": [],
        "case_references": [],
    }


# =============================================================================
# CONSOLIDATED FUNCTIONS (Reducing LLM calls)
# =============================================================================

async def checkpoint(
    query: str,
    findings: str,
    plan: str,
    client: GeminiClient,
) -> dict:
    """Combined sufficiency + replan check. Uses LITE model.

    Consolidates is_sufficient + should_replan into one call.
    Returns: {sufficient, should_replan, progress_assessment, next_steps, new_search_terms, files_to_check}
    """
    start_time = time.time()
    logger.info(f"🔍 checkpoint: evaluating investigation status")

    prompt = prompts.P_CHECKPOINT.format(
        query=query,
        findings=findings,
        plan=plan,
    )

    _log_llm_call("checkpoint", ModelTier.LITE, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.LITE)
    result = parse_json_safe(response)

    if result:
        # Coerce boolean fields to handle string values from LLM
        result["sufficient"] = _coerce_bool(result.get("sufficient", False))
        result["should_replan"] = _coerce_bool(result.get("should_replan", False))
        _log_llm_result("checkpoint", result, time.time() - start_time)
        return result

    # Fallback: not sufficient, should replan
    logger.warning("   JSON parsing failed, returning default checkpoint")
    return {
        "sufficient": False,
        "should_replan": True,
        "progress_assessment": "Unable to assess",
        "next_steps": [],
        "new_search_terms": [],
        "files_to_check": [],
    }


async def analyze_search(
    query: str,
    key_issues: list[str],
    results,  # SearchResults object
    already_read: list[str],
    client: GeminiClient,
) -> dict:
    """Combined search analysis. Uses FLASH model for strategic decisions.

    Consolidates pick_relevant_hits + analyze_results + prioritize_documents into one call.
    Returns: {relevant_hit_numbers, facts, citations, ranked_documents, additional_searches, read_deeper}
    """
    start_time = time.time()
    logger.info(f"🔍 analyze_search: combined analysis of {len(results.hits)} hits")

    # Format results for the prompt
    results_text = []
    for i, hit in enumerate(results.hits[:30], 1):  # Limit to 30 hits
        context = hit.context[:200] if hit.context else ""
        results_text.append(f"[{i}] {hit.file_path} (p.{hit.page_num}): {hit.match_text[:100]}... Context: {context}")

    prompt = prompts.P_ANALYZE_SEARCH.format(
        query=query,
        key_issues="\n".join(f"- {issue}" for issue in key_issues) if key_issues else "None identified yet",
        results="\n".join(results_text),
        already_read="\n".join(already_read) if already_read else "None",
    )

    _log_llm_call("analyze_search", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        # Extract relevant hits based on numbers returned (coerce strings to ints)
        hit_numbers = result.get("relevant_hit_numbers", [])
        relevant_hits = []
        for num in hit_numbers:
            coerced = _coerce_int(num)
            if coerced is not None and 1 <= coerced <= len(results.hits):
                relevant_hits.append(results.hits[coerced - 1])
        result["relevant_hits"] = relevant_hits

        _log_llm_result("analyze_search", f"{len(relevant_hits)} hits, {len(result.get('ranked_documents', []))} docs ranked", time.time() - start_time)
        return result

    # Fallback: return first few results to avoid stalling investigation
    logger.warning("   JSON parsing failed, using fallback (first 5 hits)")
    fallback_hits = results.hits[:5] if results.hits else []
    fallback_docs = list(set(hit.file_path for hit in fallback_hits))[:3]
    return {
        "relevant_hit_numbers": list(range(1, min(6, len(results.hits) + 1))),
        "relevant_hits": fallback_hits,
        "facts": [],
        "citations": [],
        "ranked_documents": [{"file": f, "score": 50, "criticality": "SUPPORTING"} for f in fallback_docs],
        "additional_searches": [],
        "read_deeper": fallback_docs,
    }


async def analyze_external(
    query: str,
    case_law_results: str,
    web_results: str,
    client: GeminiClient,
) -> dict:
    """Combined external research analysis. Uses FLASH model.

    Consolidates analyze_case_law_results + analyze_web_results into one call.
    Returns: {key_precedents, legal_standards, regulations, regulatory_standards, combined_framework, summary}
    """
    start_time = time.time()
    logger.info(f"🔍 analyze_external: combined analysis of case law and web results")

    prompt = prompts.P_ANALYZE_EXTERNAL.format(
        query=query,
        case_law_results=case_law_results or "No case law results found.",
        web_results=web_results or "No web/regulatory results found.",
    )

    _log_llm_call("analyze_external", ModelTier.FLASH, prompt, start_time)
    response = await client.complete(prompt, tier=ModelTier.FLASH)
    result = parse_json_safe(response)

    if result:
        _log_llm_result("analyze_external", result, time.time() - start_time)
        return result

    logger.warning("   JSON parsing failed, returning empty external analysis")
    return {
        "key_precedents": [],
        "legal_standards": [],
        "regulations": [],
        "regulatory_standards": [],
        "combined_framework": "",
        "summary": "",
    }
