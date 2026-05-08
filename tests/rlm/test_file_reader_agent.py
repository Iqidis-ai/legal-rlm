"""Tests for DocumentFileReader — structural file-reading operator."""

from __future__ import annotations

import asyncio

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation, AgentRequirement, AgentTaskView,
    DocumentFileReader, OperatorBudget,
    SubAgentDispatcher, SubAgentRegistry,
)
from irys.rlm.agents.file_reader import (
    _detect_sections, _detect_schedules, _detect_tables,
    _detect_footnotes, _detect_signature_status,
)


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


def test_detect_sections_finds_legal_numbering():
    text = """Section 4.01 Conditions Precedent

Section 4.01(a) Officer's Certificate
Section 4.01(b) Opinion of Counsel
Section 5.02 Negative Covenants
"""
    sections = _detect_sections(text)
    labels = [s["label"] for s in sections]
    assert "Section 4.01" in labels
    assert "Section 4.01(a)" in labels
    assert "Section 5.02" in labels


def test_detect_sections_finds_articles_and_markdown():
    text = """ARTICLE I — Definitions

ARTICLE II — Conditions Precedent

# Top heading
## Subsection
"""
    sections = _detect_sections(text)
    labels = [s["label"] for s in sections]
    assert any("ARTICLE I" in l for l in labels)
    assert any("ARTICLE II" in l for l in labels)
    assert "#" in labels or any(l.startswith("#") for l in labels)


def test_detect_schedules_finds_all_kinds():
    text = """See Schedule 3.01(a) for details.
The Exhibit B contains the form.
Annex II is the financial statements.
Appendix C is the agreement.
Schedule 3.01(a) is referenced again here.
"""
    schedules = _detect_schedules(text)
    labels = {s["label"].lower() for s in schedules}
    assert "schedule 3.01(a)" in labels
    assert "exhibit b" in labels
    assert "annex ii" in labels
    assert "appendix c" in labels
    # Schedule 3.01(a) appears twice but should be one entry with count=2
    sch = [s for s in schedules if s["label"].lower() == "schedule 3.01(a)"][0]
    assert sch["count"] == 2


def test_detect_tables_finds_markdown_tables():
    text = """Some prose.

| Col A | Col B |
|---|---|
| 1 | 2 |
| 3 | 4 |

More prose.

| Single | Header |
"""
    tables = _detect_tables(text)
    # Two-row table should match; one-row doesn't qualify
    assert len(tables) == 1
    assert tables[0]["n_rows"] >= 4


def test_detect_footnotes_finds_markers():
    text = "Body[1] continues. See footnote[2]: this is the body. Also [*] starred."
    fns = _detect_footnotes(text)
    markers = [f["marker"] for f in fns]
    assert "1" in markers
    assert "2" in markers


def test_detect_signature_status_signed():
    text = """IN WITNESS WHEREOF, the parties have executed this Agreement.

By: /s/ John Smith
Name: John Smith
Title: CEO
"""
    sig = _detect_signature_status(text)
    assert sig["has_signature_block"]
    assert sig["is_signed_heuristic"]


def test_detect_signature_status_unsigned():
    text = """IN WITNESS WHEREOF, the parties have executed this Agreement.

By: ___________________
Name:
Title:
"""
    sig = _detect_signature_status(text)
    assert sig["has_signature_block"]
    # No filled "/s/" or filled By:Name → unsigned heuristic
    assert sig["is_signed_heuristic"] is False


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------


def _seed_inventory(matter: MatterModel, *, doc_id: str, path: str):
    matter.inventory.upsert(
        relative_path=path,
        sha256="test_hash",
        size_bytes=1000,
        file_type="pdf",
    )


def _make_invocation(matter_id: str):
    return AgentInvocation(
        matter_id=matter_id, run_id="r",
        agent_id="reader", phase="pre_synthesis",
        persona_id=None, requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="h",
    )


def test_file_reader_no_documents_returns_success_zero_artifacts():
    m = MatterModel.open_in_memory()
    agent = DocumentFileReader()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    assert result.artifacts == ()


def test_file_reader_emits_section_map_and_schedule_artifacts():
    """Use the test hook (_test_doc_text) to feed text into the agent
    without needing a real repository."""
    m = MatterModel.open_in_memory()
    _seed_inventory(m, doc_id="d1", path="agreement.pdf")

    fixture_text = """ARTICLE I — Definitions

Section 1.01 Defined Terms

Section 4.01(a) Officer's Certificate
See Schedule 3.01(a) for details.

Section 4.01(b) Opinion of Counsel
Refer to Exhibit B.

| Term | Meaning |
|---|---|
| EBITDA | as defined in Section 1.01 |
| Permitted Lien | per Annex II |
"""

    agent = DocumentFileReader()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)

    # Patch the runtime factory to inject test text
    from irys.rlm.agents.runtime import AgentRuntime
    orig_init = AgentRuntime.__init__
    def patched_init(self, *args, **kw):
        orig_init(self, *args, **kw)
        self._test_doc_text = {"agreement.pdf": fixture_text}
    AgentRuntime.__init__ = patched_init
    try:
        out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    finally:
        AgentRuntime.__init__ = orig_init

    _, result = out.invocations[0]
    assert result.status == "success"
    kinds = {a.artifact_kind for a in result.artifacts}
    assert "document.section_map" in kinds
    assert "document.schedule_index" in kinds
    assert "document.table_index" in kinds


def test_file_reader_persists_artifacts():
    m = MatterModel.open_in_memory()
    _seed_inventory(m, doc_id="d1", path="agreement.pdf")
    fixture_text = (
        "Section 1.01 A\nSection 1.02 B\nSchedule 3.01(a) referenced.\n"
    )
    from irys.rlm.agents.runtime import AgentRuntime
    orig_init = AgentRuntime.__init__
    def patched_init(self, *args, **kw):
        orig_init(self, *args, **kw)
        self._test_doc_text = {"agreement.pdf": fixture_text}
    AgentRuntime.__init__ = patched_init
    try:
        agent = DocumentFileReader()
        reg = SubAgentRegistry(agents=(agent,))
        disp = SubAgentDispatcher(registry=reg, matter_model=m)
        asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    finally:
        AgentRuntime.__init__ = orig_init
    n = m.db.execute(
        "SELECT COUNT(*) FROM agent_artifact WHERE artifact_kind LIKE 'document.%'"
    ).fetchone()[0]
    assert n >= 2


def test_file_reader_capability_tags():
    a = DocumentFileReader()
    assert "parse.document_structure" in a.capability_tags
    assert "parse.section_map" in a.capability_tags
    assert "parse.schedule" in a.capability_tags


def test_file_reader_supports_all_five_domains():
    """Cross-domain operator per the user directive — no Harvey-LAB overfit."""
    a = DocumentFileReader()
    expected = {"legal:1", "finance:1", "coding:1", "academic_research:1", "biomedical:1"}
    assert set(a.supported_domain_profiles) == expected
