"""Standalone smoke script (no pytest).

Exercises the registry + the agent loop with canned decisions + stubbed
tool executors. Prints OK/FAIL per case. Useful while the dev env does
not have pytest installed.

Run:
    .\\venv\\Scripts\\Activate.ps1
    python tests/_smoke_research.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core import research_tools
from irys.rlm import decisions, research_agent
from irys.rlm.research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchContext,
    ResearchEmitter,
)
from irys.rlm.state import InvestigationState


def _case(name, fn):
    try:
        asyncio.run(fn()) if asyncio.iscoroutinefunction(fn) else fn()
        print(f"  OK    {name}")
        return 1, 0
    except Exception as e:
        traceback.print_exc()
        print(f"  FAIL  {name} ({e})")
        return 0, 1


def t_registry():
    assert len(research_tools.TOOL_SPECS) == 7
    names = [t.name for t in research_tools.TOOL_SPECS]
    for n in ("search_opinions", "lookup_citations", "get_opinion",
              "get_cluster", "find_citing_cases", "web_search", "fetch_url"):
        assert n in names, f"missing {n}"
    import json
    json.dumps(research_tools.tool_schemas_for_prompt())


def _mk_emitter(calls):
    async def started(s, l): calls.append(("started", l.id, l.description))
    async def update(s, lid, k, d): calls.append(("update", lid, k, dict(d)))
    async def done(s, lid): calls.append(("done", lid))
    return ResearchEmitter(started, update, done)


async def t_citation_only_batches():
    scripted = [
        {"reasoning": "batch", "actions": [{"tool": "lookup_citations",
         "args": {"text": "991 S.W.2d 849; 576 U.S. 644"}}],
         "done_after_this": False},
        {"reasoning": "done", "actions": [], "done_after_this": True},
    ]
    decisions.decide_next_action = (lambda queue=list(scripted): _make_decider(queue))()
    decisions.build_research_brief = _make_brief({"summary": "ok", "key_precedents": [],
                                                  "legal_standards": [], "regulations": [],
                                                  "combined_framework": ""})
    call_count = {"n": 0}
    async def _exec(ctx, **kw):
        call_count["n"] += 1
        return research_tools.ToolResult(
            tool="lookup_citations", args=kw, ok=True,
            data={"case_law": [{"case_name": "Trevino", "citation": "991 S.W.2d 849",
                                "source_tool": "lookup_citations", "url": "http://x",
                                "court": "tex", "validated_for_input": "991 S.W.2d 849"}]},
            update_kind="citations_validated",
            update_data={"resolved_count": 1, "unresolved_count": 0, "items": []},
            log_line="1 resolved",
        )
    research_tools.TOOLS_BY_NAME["lookup_citations"].execute = _exec

    state = InvestigationState.create(query="validate", repository_path="/tmp")
    calls = []
    agent = ResearchAgent(
        client=SimpleNamespace(),
        external_search=SimpleNamespace(courtlistener=None, tavily=None),
        emitter=_mk_emitter(calls),
        external_research_store={},
        config=ResearchAgentConfig(max_turns=3),
    )
    brief = await agent.run(state, ResearchContext(gap="g", source_path="small_repo"))
    assert call_count["n"] == 1, call_count
    assert brief["summary"] == "ok"
    assert len(state.citations) == 1 and state.citations[0].source_type == "case_law"
    kinds = [c[2] for c in calls if c[0] == "update"]
    assert "tool_call" in kinds and "citations_validated" in kinds and "analysis" in kinds


async def t_budget_exhaustion():
    decisions.decide_next_action = _make_always({
        "reasoning": "loop",
        "actions": [{"tool": "search_opinions", "args": {"q": "x"}}],
        "done_after_this": False,
    })
    decisions.build_research_brief = _make_brief({"summary": "b", "key_precedents": [],
                                                  "legal_standards": [], "regulations": [],
                                                  "combined_framework": ""})
    async def _noop(ctx, **kw):
        return research_tools.ToolResult(
            tool="search_opinions", args=kw, ok=True, data={"case_law": []},
            update_kind="external_results",
            update_data={"source": "caselaw", "count": 0, "items": []},
            log_line="noop",
        )
    research_tools.TOOLS_BY_NAME["search_opinions"].execute = _noop
    state = InvestigationState.create(query="q", repository_path="/tmp")
    calls = []
    agent = ResearchAgent(
        client=SimpleNamespace(),
        external_search=SimpleNamespace(courtlistener=None, tavily=None),
        emitter=_mk_emitter(calls),
        external_research_store={},
        config=ResearchAgentConfig(max_turns=2),
    )
    await agent.run(state, ResearchContext(gap="y"))
    tool_calls = [c for c in calls if c[0] == "update" and c[2] == "tool_call"]
    assert len(tool_calls) == 2, f"expected 2 turns, got {len(tool_calls)}"


def _make_decider(queue):
    async def _d(**_):
        return queue.pop(0) if queue else {"reasoning": "", "actions": [], "done_after_this": True}
    return _d


def _make_always(resp):
    async def _d(**_): return resp
    return _d


def _make_brief(b):
    async def _b(**_): return b
    return _b


if __name__ == "__main__":
    ok = fail = 0
    for name, fn in [("registry", t_registry),
                     ("citation_only_batches", t_citation_only_batches),
                     ("budget_exhaustion", t_budget_exhaustion)]:
        o, f = _case(name, fn)
        ok += o; fail += f
    print(f"\nSMOKE: ok={ok} fail={fail}")
    sys.exit(0 if fail == 0 else 1)
