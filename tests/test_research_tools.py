"""Tests for research tool registry + CourtListener/Tavily wrappers.

Covers tool normalization and executor contracts. The underlying HTTP
client is stubbed per test so no real network calls occur.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.research_tools import (
    TOOL_SPECS,
    TOOLS_BY_NAME,
    ToolContext,
    _normalize_cluster,
    tool_schemas_for_prompt,
)


# =============================================================================
# Fakes
# =============================================================================


class FakeCourtListenerClient:
    def __init__(self):
        self.search_calls: list[dict] = []
        self.lookup_calls: list[str] = []
        self.opinion_calls: list[dict] = []
        self.cluster_calls: list[int] = []
        self.search_return: list = []
        self.lookup_return: list = []
        self.cluster_return: dict | None = None
        self.opinion_return = None

    async def search_opinions(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_return

    async def lookup_citations(self, text, max_chars=60000):
        self.lookup_calls.append(text)
        return self.lookup_return

    async def get_cluster(self, cluster_id):
        self.cluster_calls.append(cluster_id)
        return self.cluster_return

    async def get_opinion(self, opinion_id=None, cluster_id=None, prefer="lead-opinion"):
        self.opinion_calls.append({"opinion_id": opinion_id, "cluster_id": cluster_id, "prefer": prefer})
        return self.opinion_return


def _mk_ctx(cl=None, tav=None):
    ext = SimpleNamespace(
        courtlistener=cl or FakeCourtListenerClient(),
        tavily=tav,
    )
    return ToolContext(external_search=ext)


# =============================================================================
# Registry contract
# =============================================================================


def test_registry_all_names_unique_and_have_schemas():
    names = [t.name for t in TOOL_SPECS]
    assert len(names) == len(set(names)), "duplicate tool names"
    for spec in TOOL_SPECS:
        assert spec.description, f"{spec.name} missing description"
        assert "type" in spec.parameters and spec.parameters["type"] == "object"
        assert isinstance(spec.parameters.get("properties", {}), dict)


def test_tool_schemas_for_prompt_is_json_serializable():
    import json
    schemas = tool_schemas_for_prompt()
    s = json.dumps(schemas)
    assert '"search_opinions"' in s and '"lookup_citations"' in s


# =============================================================================
# _normalize_cluster
# =============================================================================


def test_normalize_cluster_pulls_citation_from_volume_reporter_page():
    cluster = {
        "id": 123,
        "case_name": "Foo v. Bar",
        "date_filed": "2020-01-01",
        "citations": [{"volume": "991", "reporter": "S.W.2d", "page": "849"}],
        "absolute_url": "/opinion/123/",
    }
    entry = _normalize_cluster(cluster, source_tool="lookup_citations", validated_for_input="991 S.W.2d 849")
    assert entry["case_name"] == "Foo v. Bar"
    assert entry["citation"] == "991 S.W.2d 849"
    assert entry["source_tool"] == "lookup_citations"
    assert entry["validated_for_input"] == "991 S.W.2d 849"
    assert entry["url"].startswith("https://www.courtlistener.com/opinion/123/")


# =============================================================================
# lookup_citations executor
# =============================================================================


async def test_lookup_citations_resolved_and_unresolved():
    cl = FakeCourtListenerClient()
    cl.lookup_return = [
        {
            "citation": "991 S.W.2d 849",
            "status": 200,
            "clusters": [
                {"id": 111, "case_name": "Trevino v. State", "citations": [{"volume": "991", "reporter": "S.W.2d", "page": "849"}], "absolute_url": "/opinion/111/"}
            ],
        },
        {"citation": "999 X.Y.Z 000", "status": 404, "clusters": [], "error_message": "not found"},
    ]
    ctx = _mk_ctx(cl)
    res = await TOOLS_BY_NAME["lookup_citations"].execute(ctx, text="991 S.W.2d 849; 999 X.Y.Z 000")
    assert res.ok
    assert res.tool == "lookup_citations"
    assert res.update_kind == "citations_validated"
    assert res.update_data["resolved_count"] == 1
    assert res.update_data["unresolved_count"] == 1
    assert len(res.data["case_law"]) == 1
    assert res.data["case_law"][0]["validated_for_input"] == "991 S.W.2d 849"


async def test_search_opinions_returns_normalized_entries():
    cl = FakeCourtListenerClient()
    cl.search_return = [
        SimpleNamespace(
            id="12", case_name="Baz v. Qux", court="scotus",
            date_filed="2021-02-02", citation="576 U.S. 644",
            docket_number="10-1", snippet="summary",
            url="https://www.courtlistener.com/opinion/12/",
            opinion_text=None,
            to_dict=lambda: {
                "id": "12", "case_name": "Baz v. Qux", "court": "scotus",
                "date_filed": "2021-02-02", "citation": "576 U.S. 644",
                "docket_number": "10-1", "snippet": "summary",
                "url": "https://www.courtlistener.com/opinion/12/", "opinion_text": None,
            },
        )
    ]
    ctx = _mk_ctx(cl)
    res = await TOOLS_BY_NAME["search_opinions"].execute(ctx, q="due process", max_results=5)
    assert res.ok
    assert res.update_kind == "external_results"
    assert res.data["case_law"][0]["source_tool"] == "search_opinions"
    assert res.update_data["count"] == 1
