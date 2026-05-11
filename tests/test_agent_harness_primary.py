"""Primary agent-harness regression tests."""

from pathlib import Path

import pytest

from irys.core.reader import DocumentContent, PageContent
from irys.core.repository import MatterRepository
from irys.rlm.engine import InvestigationCache, RLMConfig, RLMEngine, ReadScope
from irys.rlm.state import InvestigationState


def _engine() -> RLMEngine:
    return RLMEngine(object(), config=RLMConfig(enable_external_search=False))


def test_investigation_cache_is_scope_aware():
    cache = InvestigationCache()
    prefix = ReadScope(filepath="agreement.pdf", max_chars=100_000)
    article = ReadScope(filepath="agreement.pdf", page_start=65, page_end=75, target="Article 13")

    cache.mark_extracted("agreement.pdf", scope_key=prefix.cache_key(100_000), whole_doc=True)

    assert cache.has_extracted("agreement.pdf", prefix.cache_key(100_000))
    assert not cache.has_extracted("agreement.pdf", article.cache_key(100_000))


def test_content_for_scope_reads_requested_page_range_not_prefix():
    doc = DocumentContent(
        path="agreement.pdf",
        filename="agreement.pdf",
        file_type="pdf",
        page_count=3,
        pages=[
            PageContent(1, "front matter"),
            PageContent(2, "Article 13 default clause"),
            PageContent(3, "signature page"),
        ],
        total_chars=60,
    )
    content, _ = _engine()._content_for_scope(
        doc,
        ReadScope(filepath="agreement.pdf", page_start=2, page_end=2, target="Article 13"),
    )

    assert "Article 13 default clause" in content
    assert "front matter" not in content


def test_repository_file_list_exposes_available_extraction_metadata(tmp_path: Path):
    (tmp_path / "agreement.txt").write_text("alpha\nbeta", encoding="utf-8")
    repo = MatterRepository(tmp_path)
    repo.read("agreement.txt")

    [info] = repo.get_file_list()

    assert info["filename"] == "agreement.txt"
    assert info["page_count"] == 1
    assert info["extracted_chars"] >= len("alpha\nbeta")


def test_smart_search_files_limits_to_validated_subset(tmp_path: Path):
    wanted = tmp_path / "wanted.txt"
    other = tmp_path / "other.txt"
    wanted.write_text("needle in selected file", encoding="utf-8")
    other.write_text("needle in other file", encoding="utf-8")
    repo = MatterRepository(tmp_path)

    results = repo.smart_search_files("needle", [wanted], context_lines=1)

    assert results.total_matches == 1
    assert results.hits[0].filename == "wanted.txt"


@pytest.mark.asyncio
async def test_current_fact_records_keep_provenance_and_pack_by_budget():
    engine = _engine()
    engine.config.evidence_current_facts_budget = 500
    state = InvestigationState.create("What is Article 13?", "repo")
    scope = ReadScope(filepath="agreement.pdf", page_start=65, page_end=75, target="Article 13")

    await engine._add_current_facts(
        state,
        ["Article 13 creates an event of default."],
        source_doc="agreement.pdf",
        origin="targeted_read",
        scope=scope,
    )

    packed = engine._pack_current_facts(state)

    assert "targeted_read" in packed
    assert "Article 13" in packed
    assert state.findings["current_fact_records"][0]["page_start"] == 65
