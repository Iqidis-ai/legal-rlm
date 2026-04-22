"""Research-agent tool registry.

Each tool wraps an external-search primitive (CourtListener or Tavily) and
returns a normalized :class:`ToolResult` so the agent loop can commit to
state, emit SSE events, and summarize for the research log without knowing
the underlying API.

Add a new tool by:
  1. Writing an ``async def _execute_<name>(ctx, **args)`` that returns a
     :class:`ToolResult`.
  2. Appending a :class:`ToolSpec` to :data:`TOOL_SPECS`.

Tool shape conventions on ``ToolResult.data``:
  - ``case_law``:         list[dict]   normalized case-law entries
  - ``web``:              list[dict]   normalized web results
  - ``citation_matches``: list[dict]   raw /citation-lookup/ response rows
  - ``opinion``:          dict         {id, cluster_id, case_name, text_len}
  - ``citing_cases``:     dict         {source_id, source_name, items[...]}
  - ``web_answer``:       str          optional Tavily AI answer

All entries in ``case_law`` include a ``source_tool`` provenance tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .external_search import (
    CourtListenerClient,
    ExternalSearchManager,
    LegalCase,
    TavilyClient,
)


# =============================================================================
# Shared dataclasses
# =============================================================================


@dataclass
class ToolContext:
    """Per-investigation context passed to every tool execution."""

    external_search: ExternalSearchManager
    telemetry_step: Any = None  # InvestigationStep | None — forwarded for ext_search ops


@dataclass
class ToolResult:
    """Normalized output of any tool call."""

    tool: str
    args: dict
    ok: bool = True
    error: Optional[str] = None
    data: dict = field(default_factory=dict)
    # User-facing fields (frontend consumes these)
    update_kind: str = "external_results"   # lead.update "kind"
    update_data: dict = field(default_factory=dict)
    # Dev-facing single-line log for the research log
    log_line: str = ""


@dataclass
class ToolSpec:
    """Agent-facing tool schema + binding to an executor."""

    name: str
    description: str
    parameters: dict                # JSON-schema-style {type:object, properties:{...}, required:[...]}
    execute: Callable[..., Awaitable[ToolResult]]  # (ctx, **args) -> ToolResult


# =============================================================================
# Normalization helpers (LegalCase / cluster dict -> case_law entry)
# =============================================================================


def _normalize_legal_case(
    case: LegalCase,
    source_tool: str,
    **extra: Any,
) -> dict:
    """Convert a :class:`LegalCase` dataclass into a case_law dict."""
    d = case.to_dict()
    d["source_tool"] = source_tool
    for k, v in extra.items():
        if v is not None:
            d[k] = v
    return d


def _normalize_cluster(cluster: dict, source_tool: str, **extra: Any) -> dict:
    """Convert a CourtListener cluster JSON record to a case_law dict."""
    cites = cluster.get("citations") or []
    citation_str: Optional[str] = None
    if cites:
        c0 = cites[0]
        if isinstance(c0, dict):
            parts = [str(c0.get(k, "")) for k in ("volume", "reporter", "page") if c0.get(k)]
            citation_str = " ".join(parts).strip() or None
        else:
            citation_str = str(c0)
    cid = cluster.get("id")
    abs_url = cluster.get("absolute_url") or (f"/opinion/{cid}/" if cid else "")
    entry = {
        "id": str(cid) if cid is not None else "",
        "case_name": cluster.get("case_name") or cluster.get("case_name_full") or "Unknown",
        "court": "",
        "date_filed": cluster.get("date_filed"),
        "citation": citation_str,
        "docket_number": None,
        "opinion_text": None,
        "snippet": (cluster.get("syllabus") or "")[:400] or None,
        "url": f"https://www.courtlistener.com{abs_url}" if abs_url.startswith("/") else abs_url,
        "source_tool": source_tool,
    }
    for k, v in extra.items():
        if v is not None:
            entry[k] = v
    return entry


def _case_preview(entry: dict, max_len: int = 180) -> str:
    """One-line preview used in log summaries.

    Includes cluster_id so the research agent can chain directly to
    get_cluster / get_opinion without re-running lookup_citations.
    Format: "Case Name (261 S.W.3d 316, cluster_id=12345)"
    """
    name = entry.get("case_name") or "Unknown"
    cite = entry.get("citation") or ""
    cid = entry.get("id") or ""
    parts = []
    if cite:
        parts.append(cite)
    if cid:
        parts.append(f"cluster_id={cid}")
    tail = f" ({', '.join(parts)})" if parts else ""
    s = f"{name}{tail}"
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _items_for_ui(cases: list[dict]) -> list[dict]:
    """Slim down case_law entries to the fields the frontend renders."""
    return [
        {
            "type": "caselaw",
            "name": c.get("case_name") or "Unknown",
            "citation": c.get("citation") or "",
            "snippet": (c.get("snippet") or "")[:250],
            "url": c.get("url") or "",
            "court": c.get("court") or "",
            "date": c.get("date_filed") or "",
        }
        for c in cases
    ]


# =============================================================================
# Tool executors
# =============================================================================


async def _execute_search_opinions(ctx: ToolContext, **args: Any) -> ToolResult:
    cl: CourtListenerClient = ctx.external_search.courtlistener
    q = (args.get("q") or "").strip()
    cases = await cl.search_opinions(
        query=q,
        court=args.get("court"),
        filed_after=args.get("filed_after"),
        filed_before=args.get("filed_before"),
        max_results=int(args.get("max_results", 5)),
        case_name=args.get("case_name"),
        cites_opinion_id=args.get("cites_opinion_id"),
        status=args.get("status"),
        cite_count_gte=args.get("cite_count_gte"),
        order_by=args.get("order_by", "score desc"),
        semantic=bool(args.get("semantic", False)),
        highlight=bool(args.get("highlight", False)),
    )
    entries = [_normalize_legal_case(c, source_tool="search_opinions") for c in cases]
    label = args.get("case_name") or q or "(all)"
    return ToolResult(
        tool="search_opinions",
        args=args,
        ok=True,
        data={"case_law": entries},
        update_kind="external_results",
        update_data={
            "source": "caselaw",
            "query": label,
            "count": len(entries),
            "items": _items_for_ui(entries),
        },
        log_line=f"search_opinions(q={label!r}) -> {len(entries)} results"
                 + (f"; top: {_case_preview(entries[0])}" if entries else ""),
    )


async def _execute_lookup_citations(ctx: ToolContext, **args: Any) -> ToolResult:
    cl: CourtListenerClient = ctx.external_search.courtlistener
    text = args.get("text") or ""
    rows = await cl.lookup_citations(text=text)
    resolved: list[dict] = []
    unresolved: list[dict] = []
    cluster_entries: list[dict] = []
    for row in rows:
        cite_str = row.get("citation") or (row.get("normalized_citations") or [""])[0]
        status = row.get("status")
        if status == 200 and row.get("clusters"):
            for cl_obj in row["clusters"]:
                entry = _normalize_cluster(
                    cl_obj,
                    source_tool="lookup_citations",
                    validated_for_input=cite_str,
                )
                cluster_entries.append(entry)
                resolved.append({
                    "input": cite_str,
                    "cluster_id": cl_obj.get("id"),
                    "case_name": cl_obj.get("case_name", "Unknown"),
                    "citation": entry.get("citation"),
                })
        else:
            unresolved.append({"input": cite_str, "status": status, "error": row.get("error_message", "")})
    return ToolResult(
        tool="lookup_citations",
        args={"text_chars": len(text)},
        ok=True,
        data={"case_law": cluster_entries, "citation_matches": rows},
        update_kind="citations_validated",
        update_data={
            "resolved_count": len(resolved),
            "unresolved_count": len(unresolved),
            "items": resolved[:20],
        },
        log_line=(
            f"lookup_citations({len(text)} chars) -> resolved {len(resolved)}, "
            f"unresolved {len(unresolved)}"
            + (("; " + "; ".join(_case_preview(e) for e in cluster_entries[:3])) if cluster_entries else "")
        ),
    )


async def _execute_get_opinion(ctx: ToolContext, **args: Any) -> ToolResult:
    cl: CourtListenerClient = ctx.external_search.courtlistener
    opinion_id = args.get("opinion_id")
    cluster_id = args.get("cluster_id")
    prefer = args.get("prefer", "lead-opinion")
    case = await cl.get_opinion(
        opinion_id=opinion_id,
        cluster_id=cluster_id,
        prefer=prefer,
    )
    if not case:
        return ToolResult(
            tool="get_opinion", args=args, ok=False, error="not_found",
            data={"case_law": []},
            update_kind="opinion_fetched",
            update_data={"ok": False, "cluster_id": cluster_id, "opinion_id": opinion_id},
            log_line=f"get_opinion(cluster_id={cluster_id}, opinion_id={opinion_id}) -> not found",
        )
    entry = _normalize_legal_case(case, source_tool="get_opinion")
    text_chars = len(case.opinion_text or "")
    return ToolResult(
        tool="get_opinion",
        args=args,
        ok=True,
        data={"case_law": [entry], "opinion": {
            "id": case.id,
            "cluster_id": cluster_id,
            "case_name": case.case_name,
            "text_chars": text_chars,
        }},
        update_kind="opinion_fetched",
        update_data={
            "ok": True,
            "case_name": case.case_name,
            "citation": case.citation or "",
            "url": case.url or "",
            "char_count": text_chars,
        },
        log_line=f"get_opinion({case.case_name}) -> {text_chars} chars",
    )


async def _execute_get_cluster(ctx: ToolContext, **args: Any) -> ToolResult:
    cl: CourtListenerClient = ctx.external_search.courtlistener
    cluster_id = args.get("cluster_id")
    cluster = await cl.get_cluster(cluster_id) if cluster_id is not None else None
    if not cluster:
        return ToolResult(
            tool="get_cluster", args=args, ok=False, error="not_found",
            data={"case_law": []},
            update_kind="external_results",
            update_data={"source": "caselaw", "count": 0, "items": []},
            log_line=f"get_cluster({cluster_id}) -> not found",
        )
    entry = _normalize_cluster(cluster, source_tool="get_cluster")
    return ToolResult(
        tool="get_cluster",
        args=args,
        ok=True,
        data={"case_law": [entry]},
        update_kind="external_results",
        update_data={
            "source": "caselaw",
            "query": entry.get("case_name") or f"cluster {cluster_id}",
            "count": 1,
            "items": _items_for_ui([entry]),
        },
        log_line=f"get_cluster({cluster_id}) -> {_case_preview(entry)}",
    )


async def _execute_find_citing_cases(ctx: ToolContext, **args: Any) -> ToolResult:
    """Forward-citation traversal via /search/?q=cites:<id>."""
    cl: CourtListenerClient = ctx.external_search.courtlistener
    source_id = int(args["opinion_id"])
    cases = await cl.search_opinions(
        query="",
        cites_opinion_id=source_id,
        court=args.get("court"),
        filed_after=args.get("filed_after"),
        filed_before=args.get("filed_before"),
        max_results=int(args.get("max_results", 10)),
        order_by=args.get("order_by", "dateFiled desc"),
    )
    entries = [_normalize_legal_case(c, source_tool="find_citing_cases", cites_source_id=source_id)
               for c in cases]
    return ToolResult(
        tool="find_citing_cases",
        args=args,
        ok=True,
        data={"case_law": entries,
              "citing_cases": {"source_id": source_id, "items": entries}},
        update_kind="citing_cases",
        update_data={
            "source_opinion_id": source_id,
            "count": len(entries),
            "items": _items_for_ui(entries),
        },
        log_line=f"find_citing_cases(source={source_id}) -> {len(entries)} citing cases",
    )


async def _execute_web_search(ctx: ToolContext, **args: Any) -> ToolResult:
    tav: Optional[TavilyClient] = ctx.external_search.tavily
    if tav is None or not getattr(tav, "api_key", None):
        return ToolResult(
            tool="web_search", args=args, ok=False, error="tavily_not_configured",
            data={"web": []},
            update_kind="external_results",
            update_data={"source": "web", "count": 0, "items": []},
            log_line="web_search -> tavily not configured",
        )
    q = args.get("query") or ""
    payload = await tav.search(
        query=q,
        max_results=int(args.get("max_results", 5)),
        search_depth=args.get("search_depth", "basic"),
        include_domains=args.get("include_domains"),
        include_answer=True,
    )
    raw = payload.get("results", []) if isinstance(payload, dict) else []
    entries = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score"),
            "published_date": r.get("published_date"),
            "source_type": "web",
            "source_tool": "web_search",
        }
        for r in raw
    ]
    answer = (payload or {}).get("answer") if isinstance(payload, dict) else None
    return ToolResult(
        tool="web_search",
        args=args,
        ok=True,
        data={"web": entries, "web_answer": answer},
        update_kind="external_results",
        update_data={
            "source": "web",
            "query": q,
            "count": len(entries),
            "items": [
                {"type": "web", "name": e["title"], "title": e["title"], "url": e["url"], "snippet": (e["content"] or "")[:250]}
                for e in entries
            ],
        },
        log_line=f"web_search(q={q!r}) -> {len(entries)} results"
                 + (f"; top: {entries[0]['title'][:120]}" if entries else ""),
    )


async def _execute_fetch_url(ctx: ToolContext, **args: Any) -> ToolResult:
    tav: Optional[TavilyClient] = ctx.external_search.tavily
    url = args.get("url") or ""
    if tav is None or not getattr(tav, "api_key", None) or not url:
        return ToolResult(
            tool="fetch_url", args=args, ok=False, error="unavailable",
            data={"web": []},
            update_kind="external_results",
            update_data={"source": "web", "count": 0, "items": []},
            log_line=f"fetch_url({url}) -> unavailable",
        )
    extracted = await tav.extract(urls=[url], extract_depth=args.get("extract_depth", "basic"))
    if not extracted or extracted[0].failed:
        return ToolResult(
            tool="fetch_url", args=args, ok=False, error="extract_failed",
            data={"web": []},
            update_kind="external_results",
            update_data={"source": "web", "count": 0, "items": []},
            log_line=f"fetch_url({url}) -> extract failed",
        )
    ex = extracted[0]
    entry = {
        "title": url,
        "url": ex.url or url,
        "content": ex.raw_content or "",
        "score": 1.0,
        "source_type": "web_extract",
        "source_tool": "fetch_url",
    }
    return ToolResult(
        tool="fetch_url",
        args=args,
        ok=True,
        data={"web": [entry]},
        update_kind="external_results",
        update_data={
            "source": "web",
            "query": url,
            "count": 1,
            "items": [{"type": "web", "name": entry["title"], "title": entry["title"], "url": entry["url"], "snippet": entry["content"][:250]}],
        },
        log_line=f"fetch_url({url}) -> {len(entry['content'])} chars",
    )



# =============================================================================
# Registry
# =============================================================================


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_opinions",
        description=(
            "Keyword/fielded search across CourtListener case-law opinions. "
            "Use for locating cases by topic, case name, doctrine, or with "
            "fielded operators (caseName:, cites:<id>, citeCount:[N TO *]). "
            "Prefer `lookup_citations` if you already have citation strings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Query string; may include operators."},
                "case_name": {"type": "string", "description": "Shortcut for caseName:(...)."},
                "court": {"type": "string", "description": 'Court ID, e.g. "scotus", "ca9", "tex".'},
                "filed_after": {"type": "string", "description": "YYYY-MM-DD lower bound."},
                "filed_before": {"type": "string", "description": "YYYY-MM-DD upper bound."},
                "status": {"type": "string", "description": 'e.g. "Published", "Unpublished".'},
                "cite_count_gte": {"type": "integer", "description": "Minimum citeCount filter."},
                "order_by": {"type": "string", "description": '"score desc" (default), "dateFiled desc", "citeCount desc".'},
                "max_results": {"type": "integer", "description": "Cap on results. Default 5."},
                "semantic": {"type": "boolean", "description": "Enable semantic search."},
            },
            "required": [],
        },
        execute=_execute_search_opinions,
    ),
    ToolSpec(
        name="lookup_citations",
        description=(
            "Resolve every citation inside a text blob to CourtListener clusters "
            "in ONE call (Eyecite-powered). Batch many citations together. Use "
            "whenever you have explicit reporter citations to validate."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text containing citations. Up to 60K chars. Separate multiple citations with semicolons if constructing.",
                },
            },
            "required": ["text"],
        },
        execute=_execute_lookup_citations,
    ),
    ToolSpec(
        name="get_opinion",
        description=(
            "Fetch the full text of a single opinion. Provide `cluster_id` "
            "(preferred) and optional `prefer` ('lead-opinion', 'dissent', "
            "'concurrence'), or `opinion_id` directly."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer"},
                "opinion_id": {"type": "integer"},
                "prefer": {"type": "string", "description": "Preferred sub-opinion type."},
            },
            "required": [],
        },
        execute=_execute_get_opinion,
    ),
    ToolSpec(
        name="get_cluster",
        description="Fetch a CourtListener cluster (case-level metadata) by cluster_id.",
        parameters={
            "type": "object",
            "properties": {"cluster_id": {"type": "integer"}},
            "required": ["cluster_id"],
        },
        execute=_execute_get_cluster,
    ),
    ToolSpec(
        name="find_citing_cases",
        description=(
            "List opinions that cite the given opinion (forward-citation traversal). "
            "Use to gauge an authority's influence or find later treatment."
        ),
        parameters={
            "type": "object",
            "properties": {
                "opinion_id": {"type": "integer"},
                "court": {"type": "string"},
                "filed_after": {"type": "string"},
                "filed_before": {"type": "string"},
                "max_results": {"type": "integer"},
                "order_by": {"type": "string"},
            },
            "required": ["opinion_id"],
        },
        execute=_execute_find_citing_cases,
    ),
    ToolSpec(
        name="web_search",
        description=(
            "Tavily web search. Use for current information, regulatory text, "
            "news, secondary sources, or when case law isn't what's needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "search_depth": {"type": "string", "description": '"basic" or "advanced".'},
                "include_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
        execute=_execute_web_search,
    ),
    ToolSpec(
        name="fetch_url",
        description="Extract the full text of a single web page via Tavily.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extract_depth": {"type": "string"},
            },
            "required": ["url"],
        },
        execute=_execute_fetch_url,
    ),
]


TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


def tool_schemas_for_prompt() -> list[dict]:
    """Prompt-ready schema list for the agent."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in TOOL_SPECS
    ]
