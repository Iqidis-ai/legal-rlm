"""ReAct-style external-research agent.

Drives one to a few FLASH turns that decide which research tools to call
and in what parallel batches, commits each result to the investigation
state the moment it lands, and writes a final FLASH brief consumed by
synthesis.

Integration points (kept tight so the engine owns orchestration):

- :class:`ResearchContext` — built by the caller from ``assess_small_repo``
  or the large-repo LITE gate. Carries gap, reasoning, facts, triggers.
- :class:`ResearchEmitter` — callback shim for SSE (lead started/update/done
  plus citation hook). The engine wires these to its own ``_emit_*`` helpers.
- :class:`ResearchAgent.run` — the loop. Returns the finished brief and
  leaves everything already persisted on the supplied ``state`` plus the
  ``external_research`` dict.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

from ..core.models import GeminiClient
from ..core.research_tools import (
    TOOLS_BY_NAME,
    ToolContext,
    ToolResult,
    tool_schemas_for_prompt,
)
from . import decisions
from .state import InvestigationState, Lead

if TYPE_CHECKING:
    from ..core.external_search import ExternalSearchManager
    from ..core.telemetry import InvestigationTelemetry, InvestigationStep

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & data contracts
# =============================================================================


@dataclass
class ResearchAgentConfig:
    """Budget knobs. Defaults tuned so research can't dominate an investigation."""

    max_turns: int = 4                # Hard cap on decide_next_action calls
    max_actions_per_turn: int = 6     # Soft cap; extras executed with a warning
    per_tool_timeout_s: float = 45.0  # Per-tool asyncio timeout
    turn_timeout_s: float = 90.0      # Total timeout per turn (decide + dispatch)


@dataclass
class ResearchContext:
    """Inputs the caller supplies to the agent."""

    gap: str = ""
    reasoning: str = ""
    cached_facts: list[str] = field(default_factory=list)
    triggers_summary: str = ""
    source_path: str = "unknown"     # "small_repo" | "large_repo"


# User-facing message helpers: deliberately short / non-technical.
_KIND_USER_MESSAGES = {
    "tool_call":           "Planning next research calls",
    "external_results":    "Found external results",
    "citations_validated": "Validated case citations",
    "opinion_fetched":     "Fetched full opinion text",
    "citing_cases":        "Found cases citing this authority",
    "analysis":            "External research analysis",
}


@dataclass
class ResearchEmitter:
    """Thin wrapper around the engine's SSE helpers."""

    emit_lead_started: Callable[[InvestigationState, Lead], Awaitable[None]]
    emit_lead_update: Callable[[InvestigationState, str, str, dict], Awaitable[None]]
    emit_lead_done: Callable[[InvestigationState, str], Awaitable[None]]
    on_citation: Optional[Callable[[Any], None]] = None


# =============================================================================
# Rolling research log (structured, compact; reasoning is discarded turn-to-turn)
# =============================================================================


@dataclass
class _LogEntry:
    turn: int
    tool: str
    args_preview: str
    summary: str


class ResearchLog:
    """Dev-facing rolling summary the agent consumes on every turn.

    Contains tool calls + one-line result summaries only — prior reasoning
    is NOT fed back, which keeps the prompt bounded regardless of turn count.
    """

    def __init__(self) -> None:
        self._entries: list[_LogEntry] = []

    def append(self, turn: int, tool: str, args: dict, summary: str) -> None:
        args_preview = ", ".join(
            f"{k}={_shorten(v)}" for k, v in args.items() if v is not None
        )[:280]
        self._entries.append(_LogEntry(turn, tool, args_preview, summary))

    def render(self, max_entries: int = 40) -> str:
        entries = self._entries[-max_entries:]
        if not entries:
            return ""
        return "\n".join(
            f"T{e.turn} {e.tool}({e.args_preview}) -> {e.summary}" for e in entries
        )


def _shorten(v: Any, limit: int = 80) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _format_case_law_for_brief(case_law: list[dict], max_cases: int = 100) -> str:
    """Format case law entries for the brief-builder (FLASH) LLM.

    Entries from ``get_opinion`` carry a full ``opinion_text`` and get a
    generous limit so the model can actually extract holdings.  Entries from
    search tools carry only a short ``snippet`` and stay concise.
    """
    if not case_law:
        return ""
    lines = []
    for c in case_law[:max_cases]:
        opinion_text = c.get("opinion_text")
        snippet = c.get("snippet")
        if opinion_text:
            # Full opinion fetch — pass substantial text to the brief builder.
            content = opinion_text[:15_000]
            label = "Opinion text"
        elif snippet:
            content = snippet[:400]
            label = "Snippet"
        else:
            content = "No content available"
            label = "Note"
        lines.append(
            f"- **{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
            f"  Court: {c.get('court') or 'Unknown'} | Date: {c.get('date_filed') or 'Unknown'}"
            f" | Source: {c.get('source_tool') or 'unknown'}\n"
            f"  {label}: {content}"
        )
    return "\n\n".join(lines)


def _format_last_turn_results(pairs: list[tuple[dict, Any]]) -> str:
    """Produce a structured, human-readable summary of one turn's tool results.

    Shown to the decision LLM at the *start* of the next turn so it can
    reason about what was just retrieved without seeing the full research log.

    Per-tool content rules (Option B):
    - ``get_opinion``        : metadata + first 800 chars of opinion text
    - ``get_cluster``        : metadata only (no text in cluster records)
    - ``lookup_citations``   : resolved case names + cluster_ids (from log_line)
    - ``search_opinions`` /
      ``find_citing_cases``  : list of case names + citations
    - ``web_search`` /
      ``fetch_url``          : list of titles + URLs
    """
    parts: list[str] = []
    for action, res in pairs:
        tool = action.get("tool", "?")
        if isinstance(res, Exception) or not isinstance(res, ToolResult):
            parts.append(f"[{tool}] ERROR")
            continue
        if not res.ok:
            parts.append(f"[{tool}] FAILED: {res.error or 'unknown error'}")
            continue

        if tool == "get_opinion":
            entries = res.data.get("case_law", [])
            if entries:
                e = entries[0]
                preview = (e.get("opinion_text") or "")[:800]
                parts.append(
                    f"[get_opinion] {e.get('case_name') or 'Unknown'}"
                    f" ({e.get('citation') or 'N/A'})\n"
                    f"  Court: {e.get('court') or 'N/A'}"
                    f" | Date: {e.get('date_filed') or 'N/A'}\n"
                    f"  Text preview:\n{preview}"
                )

        elif tool in ("search_opinions", "find_citing_cases"):
            entries = res.data.get("case_law", [])
            if entries:
                items = "; ".join(
                    f"{e.get('case_name') or 'Unknown'}"
                    + (f" ({e.get('citation')})" if e.get("citation") else "")
                    + (f" [cluster_id={e.get('id')}]" if e.get("id") else "")
                    for e in entries
                )
                parts.append(f"[{tool}] Found {len(entries)} case(s): {items}")

        elif tool == "get_cluster":
            entries = res.data.get("case_law", [])
            if entries:
                e = entries[0]
                parts.append(
                    f"[get_cluster] {e.get('case_name') or 'Unknown'}"
                    f" ({e.get('citation') or 'N/A'})"
                    f" cluster_id={e.get('id') or 'N/A'}"
                    f" | Court: {e.get('court') or 'N/A'}"
                    f" | Date: {e.get('date_filed') or 'N/A'}"
                )

        elif tool == "lookup_citations":
            # log_line already contains cluster_ids from _case_preview
            parts.append(f"[lookup_citations] {res.log_line}")

        elif tool in ("web_search", "fetch_url"):
            entries = res.data.get("web", [])
            if entries:
                titles = "; ".join(
                    e.get("title") or e.get("url") or "?" for e in entries[:5]
                )
                parts.append(f"[{tool}] {len(entries)} result(s): {titles}")

    return "\n\n".join(parts)


def _format_web_for_brief(web: list[dict], web_answer: Optional[str], max_items: int = 30) -> str:
    if not web and not web_answer:
        return ""
    lines = []
    if web_answer:
        lines.append(f"**Summary:** {web_answer}")
    for r in (web or [])[:max_items]:
        lines.append(
            f"- **{r.get('title', 'Untitled')}**\n"
            f"  URL: {r.get('url', '')}\n"
            f"  Content: {(r.get('content') or '')[:300]}"
        )
    return "\n\n".join(lines)


# =============================================================================
# Agent
# =============================================================================


class ResearchAgent:
    """Tool-calling research agent. Safe to instantiate per investigation."""

    def __init__(
        self,
        client: GeminiClient,
        external_search: "ExternalSearchManager",
        emitter: ResearchEmitter,
        external_research_store: dict,
        config: Optional[ResearchAgentConfig] = None,
        telemetry: Optional["InvestigationTelemetry"] = None,
    ) -> None:
        self.client = client
        self.external_search = external_search
        self.emitter = emitter
        self.store = external_research_store
        self.config = config or ResearchAgentConfig()
        self.telemetry = telemetry

        # In-run cache keyed by (tool, normalized_args) to avoid duplicate calls.
        self._call_cache: dict[tuple, ToolResult] = {}
        # Tracks token-sets of query-like args per tool for near-duplicate detection.
        self._prior_queries: dict[str, list[set[str]]] = {}

    # ---- Public entry point --------------------------------------------------

    async def run(
        self,
        state: InvestigationState,
        context: ResearchContext,
    ) -> dict:
        """Run the loop. Returns the brief; also persists it to the store."""
        logger.info(
            "ResearchAgent.run: source=%s gap=%r max_turns=%d",
            context.source_path, context.gap[:80], self.config.max_turns,
        )
        log = ResearchLog()
        tool_schemas = tool_schemas_for_prompt()
        # Structured summary of the most recent turn's results, passed to the
        # next decide_next_action call so the model can reason about what it
        # just retrieved without relying solely on the one-line log entries.
        last_turn_content: str = ""

        for turn in range(1, self.config.max_turns + 1):
            try:
                decision = await asyncio.wait_for(
                    decisions.decide_next_action(
                        query=state.query,
                        gap=context.gap,
                        reasoning=context.reasoning,
                        facts=context.cached_facts,
                        triggers_summary=context.triggers_summary,
                        tool_schemas=tool_schemas,
                        research_log=log.render(),
                        last_turn_content=last_turn_content,
                        client=self.client,
                        active_step=self._begin_step(f"research_turn_{turn}"),
                    ),
                    timeout=self.config.turn_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning("decide_next_action timed out on turn %d", turn)
                break
            except Exception as e:
                logger.warning("decide_next_action failed on turn %d: %s", turn, e)
                break

            actions = decision.get("actions") or []
            reasoning = decision.get("reasoning") or ""
            done = bool(decision.get("done_after_this"))

            if actions:
                await self._announce_turn(state, turn, reasoning, actions)
                last_turn_content = await self._dispatch_turn(state, turn, actions, log)
            else:
                last_turn_content = ""

            if done or not actions:
                logger.info("research loop exiting at turn %d (done=%s, actions=%d)",
                            turn, done, len(actions))
                break

        # Always build the brief, even on empty / parse-failed loops.
        brief = await self._build_brief(state)
        return brief

    # ---- Turn lifecycle ------------------------------------------------------

    async def _announce_turn(
        self,
        state: InvestigationState,
        turn: int,
        reasoning: str,
        actions: list[dict],
    ) -> None:
        lead = Lead.create(
            description=f"Research step {turn}",
            source="research_agent",
        )
        lead.lead_type = "research_turn"
        state.leads.append(lead)
        await self.emitter.emit_lead_started(state, lead)
        # User-friendly message string + dev-friendly reasoning payload.
        actions_preview = [
            {"tool": a.get("tool"), "args": a.get("args") or {}}
            for a in actions[: self.config.max_actions_per_turn]
        ]
        await self.emitter.emit_lead_update(
            state, lead.id, "tool_call",
            {
                "message": _KIND_USER_MESSAGES["tool_call"],
                "turn": turn,
                "reasoning": reasoning,
                "actions": actions_preview,
                "action_count": len(actions),
            },
        )
        # Stash lead id on the turn context so dispatch can attach updates.
        self._current_turn_lead_id = lead.id

    async def _dispatch_turn(
        self,
        state: InvestigationState,
        turn: int,
        actions: list[dict],
        log: ResearchLog,
    ) -> str:
        """Execute all actions for one turn in parallel.

        Returns a structured summary of this turn's results so the *next*
        ``decide_next_action`` call can reason about what was just retrieved
        (Option B: metadata + short extract for opinion fetches).
        """
        if len(actions) > self.config.max_actions_per_turn:
            logger.warning("turn %d: %d actions exceeds soft cap %d; executing anyway",
                           turn, len(actions), self.config.max_actions_per_turn)

        ctx = ToolContext(
            external_search=self.external_search,
            telemetry_step=None,
        )
        tasks = [self._call_tool(ctx, action) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pairs: list[tuple[dict, Any]] = []
        for action, res in zip(actions, results):
            tool_name = action.get("tool", "?")
            args = action.get("args") or {}
            if isinstance(res, Exception):
                logger.warning("tool %s failed: %s", tool_name, res)
                log.append(turn, tool_name, args, f"ERROR: {str(res)[:120]}")
                pairs.append((action, res))
                continue
            assert isinstance(res, ToolResult)
            await self._commit_result(state, res)
            summary = res.log_line or "(no summary)"
            if self._is_near_duplicate(tool_name, args):
                summary = f"[NEAR-DUPLICATE of prior call] {summary}"
            log.append(turn, tool_name, args, summary)
            self._record_for_dedup(tool_name, args)
            pairs.append((action, res))

        lead_id = getattr(self, "_current_turn_lead_id", None)
        if lead_id:
            await self.emitter.emit_lead_done(state, lead_id)
            self._current_turn_lead_id = None

        return _format_last_turn_results(pairs)


    # ---- Tool execution -----------------------------------------------------

    async def _call_tool(self, ctx: ToolContext, action: dict) -> ToolResult:
        name = action.get("tool") or ""
        args = dict(action.get("args") or {})
        spec = TOOLS_BY_NAME.get(name)
        if spec is None:
            return ToolResult(
                tool=name, args=args, ok=False, error="unknown_tool",
                data={}, update_kind="external_results",
                update_data={"source": "caselaw", "count": 0, "items": []},
                log_line=f"unknown_tool({name})",
            )

        cache_key = self._cache_key(name, args)
        if cache_key in self._call_cache:
            # Surface the duplicate to the model instead of silently returning cache.
            # This gives decide_next_action a visible [DUPLICATE] signal to stop looping
            # on near-identical queries (e.g. Tavily "overruled" / "abrogated" variants).
            logger.info("tool %s duplicate call suppressed", name)
            return ToolResult(
                tool=name, args=args, ok=True, error=None,
                data={}, update_kind="external_results",
                update_data={
                    "source": "caselaw",
                    "count": 0,
                    "items": [],
                    "skipped_duplicate": True,
                },
                log_line=f"[DUPLICATE] {name}(...) — already executed this turn-set, skipped",
            )

        try:
            res = await asyncio.wait_for(
                spec.execute(ctx, **args),
                timeout=self.config.per_tool_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("tool %s timed out", name)
            return ToolResult(
                tool=name, args=args, ok=False, error="timeout",
                data={}, update_kind="external_results",
                update_data={"source": "caselaw", "count": 0, "items": []},
                log_line=f"{name}(...) -> timeout",
            )
        except Exception as e:
            logger.warning("tool %s raised %s", name, e)
            return ToolResult(
                tool=name, args=args, ok=False, error=str(e)[:160],
                data={}, update_kind="external_results",
                update_data={"source": "caselaw", "count": 0, "items": []},
                log_line=f"{name}(...) -> ERROR {str(e)[:120]}",
            )

        self._call_cache[cache_key] = res
        return res

    @staticmethod
    def _cache_key(name: str, args: dict) -> tuple:
        def _norm(v: Any) -> Any:
            if isinstance(v, list):
                return tuple(_norm(x) for x in v)
            if isinstance(v, dict):
                return tuple(sorted((k, _norm(x)) for k, x in v.items()))
            return v
        return (name, tuple(sorted((k, _norm(v)) for k, v in args.items())))

    # ---- Near-duplicate query detection (soft signal, log-only) --------------

    _QUERY_ARG_KEYS = ("query", "q", "case_name", "text")

    @classmethod
    def _tokenize_query_args(cls, args: dict) -> Optional[set[str]]:
        """Extract a lowercase token set from any query-like string arg."""
        parts: list[str] = []
        for k in cls._QUERY_ARG_KEYS:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        if not parts:
            return None
        blob = " ".join(parts).lower()
        tokens = {t for t in blob.replace('"', " ").split() if len(t) > 2}
        return tokens or None

    def _is_near_duplicate(self, tool: str, args: dict, threshold: float = 0.6) -> bool:
        tokens = self._tokenize_query_args(args)
        if not tokens:
            return False
        for prior in self._prior_queries.get(tool, []):
            if not prior:
                continue
            overlap = len(tokens & prior) / max(len(tokens | prior), 1)
            if overlap >= threshold:
                return True
        return False

    def _record_for_dedup(self, tool: str, args: dict) -> None:
        tokens = self._tokenize_query_args(args)
        if not tokens:
            return
        self._prior_queries.setdefault(tool, []).append(tokens)

    # ---- Commit to state + emit SSE -----------------------------------------

    async def _commit_result(self, state: InvestigationState, res: ToolResult) -> None:
        """Write tool result into external_research store and state.citations, then emit.

        Commits happen BEFORE the next decision turn, so partial results
        survive timeouts or loop aborts.
        """
        data = res.data or {}
        # 1) case_law
        for entry in data.get("case_law", []) or []:
            self.store.setdefault("case_law", []).append(entry)
            self._add_case_law_citation(state, entry)
        # 2) citation-lookup raw matches (inspection-only slot)
        if "citation_matches" in data:
            self.store.setdefault("citation_matches", []).extend(data["citation_matches"])
        # 3) opinion full-text slot
        if "opinion" in data and data["opinion"]:
            op = data["opinion"]
            self.store.setdefault("opinions", {})[str(op.get("id"))] = op
        # 4) citing-cases inspection slot
        if "citing_cases" in data and data["citing_cases"]:
            cc = data["citing_cases"]
            key = str(cc.get("source_id"))
            self.store.setdefault("citing_cases", {})[key] = cc
        # 5) web
        for entry in data.get("web", []) or []:
            self.store.setdefault("web", []).append(entry)
            self._add_web_citation(state, entry)
        # 6) tavily answer passthrough (retain earliest non-null)
        if data.get("web_answer") and not self.store.get("web_answer"):
            self.store["web_answer"] = data["web_answer"]

        # Emit SSE on whichever lead id the current turn was announced under.
        lead_id = getattr(self, "_current_turn_lead_id", None)
        if not lead_id:
            return
        update = dict(res.update_data or {})
        update.setdefault("message", _KIND_USER_MESSAGES.get(res.update_kind, "Research update"))
        update["tool"] = res.tool
        update["ok"] = res.ok
        if not res.ok and res.error:
            update["error"] = res.error
        await self.emitter.emit_lead_update(state, lead_id, res.update_kind, update)

    def _add_case_law_citation(self, state: InvestigationState, entry: dict) -> None:
        name = entry.get("case_name") or "Unknown Case"
        doc_name = f"[Case Law] {name}"
        # Truncate to a short excerpt for the UI citation list.
        # The full opinion_text stays in store["case_law"] for the brief builder.
        raw_text = (entry.get("opinion_text") or entry.get("snippet") or "")
        citation = state.add_citation(
            document=doc_name,
            page=None,
            text=raw_text[:500],
            context=f"Citation: {entry.get('citation') or 'N/A'} | Court: {entry.get('court') or 'N/A'}",
            relevance=(
                f"Validated via citation lookup (input: {entry.get('validated_for_input')})"
                if entry.get("validated_for_input") else "External case law research"
            ),
            url=entry.get("url"),
            mime=entry.get("mime"),
            source_type="case_law",
        )
        if citation and self.emitter.on_citation:
            self.emitter.on_citation(citation)

    def _add_web_citation(self, state: InvestigationState, entry: dict) -> None:
        title = entry.get("title") or "Unknown Source"
        citation = state.add_citation(
            document=f"[Web] {title}",
            page=None,
            text=entry.get("content") or "",
            context=f"URL: {entry.get('url') or 'N/A'}",
            relevance="External regulatory / web research",
            url=entry.get("url"),
            mime=entry.get("mime"),
            source_type="web",
        )
        if citation and self.emitter.on_citation:
            self.emitter.on_citation(citation)

    # ---- Final brief --------------------------------------------------------

    async def _build_brief(self, state: InvestigationState) -> dict:
        """Run build_research_brief and persist into the analysis slot."""
        case_law_text = _format_case_law_for_brief(self.store.get("case_law", []))
        web_text = _format_web_for_brief(self.store.get("web", []), self.store.get("web_answer"))

        step = self._begin_step("research_brief")
        t0 = time.monotonic()
        try:
            brief = await decisions.build_research_brief(
                query=state.query,
                case_law_results=case_law_text,
                web_results=web_text,
                client=self.client,
                active_step=step,
            )
        except Exception as e:
            logger.warning("build_research_brief failed: %s", e)
            brief = {"key_precedents": [], "legal_standards": [], "regulations": [],
                     "combined_framework": "", "summary": ""}
        finally:
            self._end_step(step)

        self.store["analysis"] = brief

        # Announce as its own lead so the frontend can display it distinctly.
        analysis_lead = Lead.create(
            description="External research analysis",
            source="research_agent",
        )
        analysis_lead.lead_type = "analysis"
        state.leads.append(analysis_lead)
        await self.emitter.emit_lead_started(state, analysis_lead)
        await self.emitter.emit_lead_update(
            state, analysis_lead.id, "analysis",
            {
                "message": _KIND_USER_MESSAGES["analysis"],
                "summary": brief.get("summary", ""),
                "key_precedents": brief.get("key_precedents", []),
                "legal_standards": brief.get("legal_standards", []),
                "regulations": brief.get("regulations", []),
                "combined_framework": brief.get("combined_framework", ""),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        await self.emitter.emit_lead_done(state, analysis_lead.id)
        return brief

    # ---- Telemetry helpers --------------------------------------------------

    def _begin_step(self, name: str) -> Optional["InvestigationStep"]:
        if self.telemetry is None:
            return None
        try:
            return self.telemetry.begin_step(name, "research_agent")
        except Exception:
            return None

    def _end_step(self, step: Optional["InvestigationStep"]) -> None:
        if self.telemetry is None or step is None:
            return
        try:
            self.telemetry.end_step(step)
        except Exception:
            pass
