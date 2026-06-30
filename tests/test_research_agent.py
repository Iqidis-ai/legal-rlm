"""Agent-loop tests for ResearchAgent.

The ``decisions.decide_next_action`` + ``decisions.build_research_brief``
functions are monkeypatched with canned responses so no LLM calls are
made. Each tool's executor is stubbed in :mod:`irys.core.research_tools`
so no HTTP calls occur.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


# =============================================================================
# Shared fixtures / fakes
# =============================================================================


def _mk_state(query: str = "Validate 991 S.W.2d 849") -> InvestigationState:
    return InvestigationState.create(query=query, repository_path="/tmp/fake")


async def _no_emit(*args, **kwargs):
    return None


def _mk_emitter(calls: list[tuple]) -> ResearchEmitter:
    async def started(state, lead):
        calls.append(("started", lead.id, lead.description))

    async def update(state, lead_id, kind, data):
        calls.append(("update", lead_id, kind, dict(data)))

    async def done(state, lead_id):
        calls.append(("done", lead_id))

    return ResearchEmitter(emit_lead_started=started, emit_lead_update=update, emit_lead_done=done)


class _FakeClient:
    """Placeholder — monkeypatches bypass it."""


def _patch_decide(monkeypatch, scripted: list[dict]):
    """Replace decide_next_action with a queue-based responder."""
    queue = list(scripted)

    async def _stub(**kwargs):
        if queue:
            return queue.pop(0)
        return {"reasoning": "default done", "actions": [], "done_after_this": True}

    monkeypatch.setattr(decisions, "decide_next_action", _stub)


def _patch_brief(monkeypatch, brief: dict | None = None):
    brief = brief or {"key_precedents": [], "legal_standards": [], "regulations": [],
                      "combined_framework": "stub", "summary": "stub summary"}

    async def _stub(**kwargs):
        return brief

    monkeypatch.setattr(decisions, "build_research_brief", _stub)


def _stub_tool_executor(monkeypatch, name: str, result_fn):
    """Swap a single tool executor for a test-controlled version."""
    spec = research_tools.TOOLS_BY_NAME[name]
    monkeypatch.setattr(spec, "execute", result_fn)


# =============================================================================
# Tests
# =============================================================================


async def test_citation_only_path_batches_into_one_lookup(monkeypatch):
    _patch_decide(monkeypatch, [
        {"reasoning": "batch", "actions": [
            {"tool": "lookup_citations", "args": {"text": "991 S.W.2d 849; 576 U.S. 644"}}
        ], "done_after_this": False},
        {"reasoning": "done", "actions": [], "done_after_this": True},
    ])
    _patch_brief(monkeypatch)

    call_count = {"n": 0}
    async def _exec_lookup(ctx, **kwargs):
        call_count["n"] += 1
        return research_tools.ToolResult(
            tool="lookup_citations", args=kwargs, ok=True,
            data={"case_law": [
                {"case_name": "Trevino v. State", "citation": "991 S.W.2d 849",
                 "source_tool": "lookup_citations", "url": "http://x", "court": "tex",
                 "validated_for_input": "991 S.W.2d 849"},
            ]},
            update_kind="citations_validated",
            update_data={"resolved_count": 1, "unresolved_count": 0, "items": []},
            log_line="lookup_citations -> 1 resolved",
        )
    _stub_tool_executor(monkeypatch, "lookup_citations", _exec_lookup)

    state = _mk_state()
    store: dict = {}
    calls: list[tuple] = []
    agent = ResearchAgent(
        client=_FakeClient(),
        external_search=SimpleNamespace(courtlistener=None, tavily=None),
        emitter=_mk_emitter(calls),
        external_research_store=store,
        config=ResearchAgentConfig(max_turns=3),
    )
    brief = await agent.run(state, ResearchContext(gap="validate", source_path="small_repo"))

    assert call_count["n"] == 1, "lookup_citations should run exactly once"
    assert brief.get("summary") == "stub summary"
    assert len(store["case_law"]) == 1
    assert len(state.citations) == 1
    assert state.citations[0].source_type == "case_law"
    kinds = [c[2] for c in calls if c[0] == "update"]
    assert "tool_call" in kinds and "citations_validated" in kinds and "analysis" in kinds


async def test_parse_failure_exits_loop_and_still_builds_brief(monkeypatch):
    async def _stub(**_):
        return {"reasoning": "parse_failure", "actions": [], "done_after_this": True}
    monkeypatch.setattr(decisions, "decide_next_action", _stub)
    _patch_brief(monkeypatch, {"summary": "ok", "key_precedents": [], "legal_standards": [],
                               "regulations": [], "combined_framework": ""})

    state = _mk_state()
    calls: list[tuple] = []
    agent = ResearchAgent(
        client=_FakeClient(),
        external_search=SimpleNamespace(courtlistener=None, tavily=None),
        emitter=_mk_emitter(calls),
        external_research_store={},
    )
    brief = await agent.run(state, ResearchContext(gap="x"))
    assert brief["summary"] == "ok"
    # No tool_call emitted (no actions), but analysis still fires.
    assert any(c[0] == "update" and c[2] == "analysis" for c in calls)


async def test_budget_exhaustion_still_builds_brief(monkeypatch):
    # Decide always returns one no-op action that the tool no-op executes.
    async def _stub(**_):
        return {"reasoning": "loop",
                "actions": [{"tool": "search_opinions", "args": {"q": "x"}}],
                "done_after_this": False}
    monkeypatch.setattr(decisions, "decide_next_action", _stub)
    _patch_brief(monkeypatch)

    async def _noop(ctx, **kwargs):
        return research_tools.ToolResult(
            tool="search_opinions", args=kwargs, ok=True, data={"case_law": []},
            update_kind="external_results",
            update_data={"source": "caselaw", "count": 0, "items": []},
            log_line="noop",
        )
    _stub_tool_executor(monkeypatch, "search_opinions", _noop)

    state = _mk_state()
    calls: list[tuple] = []
    agent = ResearchAgent(
        client=_FakeClient(),
        external_search=SimpleNamespace(courtlistener=None, tavily=None),
        emitter=_mk_emitter(calls),
        external_research_store={},
        config=ResearchAgentConfig(max_turns=2),
    )
    brief = await agent.run(state, ResearchContext(gap="y"))
    assert brief["summary"] == "stub summary"
    # Two tool_call turns (max_turns=2), then analysis.
    tool_calls = [c for c in calls if c[0] == "update" and c[2] == "tool_call"]
    assert len(tool_calls) == 2
