from pathlib import Path

import pytest

from irys.core.repository import FileInfo
from irys.rlm.engine import RLMConfig, RLMEngine
from irys.rlm.state import InvestigationState


class _StubClient:
    pass


class _NoLLMClient:
    async def complete(self, *args, **kwargs):
        raise AssertionError("inventory count query should not call an LLM")


def _file(relative_path: str) -> FileInfo:
    path = Path(relative_path)
    return FileInfo(
        path=path,
        filename=path.name,
        file_type=path.suffix.lower(),
        size_bytes=100,
        relative_path=relative_path,
    )


def test_initial_deep_read_selection_prioritizes_finance_filings_by_path():
    engine = RLMEngine(
        gemini_client=_StubClient(),
        config=RLMConfig(max_initial_deep_read_documents=3),
    )
    state = InvestigationState.create(
        "How have Datadog financials changed over time?",
        ".",
    )
    files = [
        _file("archive/press/product-launch.pdf"),
        _file("archive/sec/10-K/2023/ddog-2023-10-k.pdf"),
        _file("archive/sec/10-Q/2024/q2/ddog-2024-q2-10-q.pdf"),
        _file("archive/sec/S-8/2024/equity-plan.pdf"),
        _file("archive/investor-relations/EX-99/q4-earnings-release.pdf"),
    ]

    selected = engine._select_initial_deep_read_files(state, files)
    selected_paths = {item.relative_path for item in selected}

    assert selected_paths == {
        "archive/sec/10-K/2023/ddog-2023-10-k.pdf",
        "archive/sec/10-Q/2024/q2/ddog-2024-q2-10-q.pdf",
        "archive/investor-relations/EX-99/q4-earnings-release.pdf",
    }
    selection = state.findings["initial_deep_read_selection"]
    assert selection["mode"] == "path_scored"
    assert selection["skipped_count"] == 2
    assert selection["finance_focused"] is True


def test_initial_deep_read_selection_keeps_orientation_targets():
    engine = RLMEngine(
        gemini_client=_StubClient(),
        config=RLMConfig(max_initial_deep_read_documents=2),
    )
    state = InvestigationState.create("Analyze damages trend", ".")
    state.findings["initial_plan"] = {
        "target_documents": ["misc/random-note.pdf"],
    }
    files = [
        _file("archive/sec/10-K/2023/ddog-2023-10-k.pdf"),
        _file("archive/sec/10-Q/2024/q2/ddog-2024-q2-10-q.pdf"),
        _file("misc/random-note.pdf"),
    ]

    selected = engine._select_initial_deep_read_files(state, files)
    selected_paths = {item.relative_path for item in selected}

    assert "misc/random-note.pdf" in selected_paths
    assert len(selected_paths) == 2


def test_simple_lookup_path_match_outranks_bad_orientation_target():
    engine = RLMEngine(
        gemini_client=_StubClient(),
        config=RLMConfig(max_initial_deep_read_documents=1),
    )
    state = InvestigationState.create("when did Kumail join data dog?", ".")
    state.findings["initial_plan"] = {
        "target_documents": ["filings/ir/landing/corporate-governance_governance-overview.pdf"],
    }
    files = [
        _file("filings/ir/landing/corporate-governance_governance-overview.pdf"),
        _file("news/kumail-nanjiani-join-datadogs-dash-conference-featured-speaker.pdf"),
    ]

    selected = engine._select_initial_deep_read_files(state, files)

    assert [item.relative_path for item in selected] == [
        "news/kumail-nanjiani-join-datadogs-dash-conference-featured-speaker.pdf",
    ]
    selection = state.findings["initial_deep_read_selection"]
    assert selection["simple_lookup_terms"] == ["kumail"]


def test_initial_deep_read_selection_reads_all_small_repos():
    engine = RLMEngine(
        gemini_client=_StubClient(),
        config=RLMConfig(max_initial_deep_read_documents=20),
    )
    state = InvestigationState.create("Analyze the contract", ".")
    files = [
        _file("contracts/msa.pdf"),
        _file("invoices/march-invoice.pdf"),
    ]

    selected = engine._select_initial_deep_read_files(state, files)

    assert [item.relative_path for item in selected] == [
        "contracts/msa.pdf",
        "invoices/march-invoice.pdf",
    ]
    assert state.findings["initial_deep_read_selection"]["mode"] == "all_files"


def test_inventory_fast_path_does_not_hijack_content_questions():
    assert RLMEngine._repository_inventory_target(
        "list the risk factors in the 10-K",
    ) is None
    assert RLMEngine._repository_inventory_target(
        "how many 10-Ks mention revenue?",
    ) is None


@pytest.mark.asyncio
async def test_inventory_count_query_returns_before_profiling(tmp_path):
    ten_k_dir = tmp_path / "filings" / "sec" / "10-K"
    ten_k_dir.mkdir(parents=True)
    for year in range(2018, 2025):
        (ten_k_dir / f"ddog-{year}-10-k.pdf").write_text("", encoding="utf-8")
    other_dir = tmp_path / "filings" / "sec" / "S-8"
    other_dir.mkdir(parents=True)
    (other_dir / "equity-plan.pdf").write_text("", encoding="utf-8")

    engine = RLMEngine(
        gemini_client=_NoLLMClient(),
        config=RLMConfig(enable_matter_model=False),
    )

    state = await engine.investigate(
        "how many 10ks are there?",
        tmp_path,
        research_mode="simple",
    )

    assert state.status == "completed"
    assert "There are 7 10-K documents" in state.findings["final_output"]
    assert state.findings["repository_inventory_answer"]["count"] == 7
    assert state.findings.get("all_documents_ingested") is None
    assert state.documents_read == 0
