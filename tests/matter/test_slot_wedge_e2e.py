"""End-to-end smoke tests for the slot wedge.

Exercises:
- _profile_dataset_shape() actually registers slots from a corpus
  containing MSA mentions (catches the DocumentContent type bug from
  the round-1 adversarial audit).
- The market_row deep-read parsing path writes typed_evidence and
  marks the matching slot filled.
- _build_extraction_slot_row_summary emits a markdown table when slots
  are filled, and empty string otherwise.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from irys.matter import MatterModel
from irys.rlm.engine import RLMEngine
from irys.rlm.state import InvestigationState


REGULATORY_DOC = """
Market Concentration Analysis — Industrial Gases Acquisition Review

This memo summarizes regional market concentration for the proposed Meridian/PeakAir
acquisition. Key markets analyzed:

- Greenville-Spartanburg MSA: pre-merger HHI 2234, post-merger HHI 3224 (delta 990).
  Acquirer share 28%, target share 18%. Structural presumption triggered.
- Savannah MSA: HHI moves from 2234 to 3224 with delta 990. Combined share above 40%.
- Charleston MSA: post-merger HHI 3082, delta 1092. Above presumption threshold.
- Atlanta MSA: HHI 2681 post-merger, delta 836. Triggers structural presumption.
- Birmingham MSA: HHI 2626, delta 864. Combined share elevated.
- Charlotte MSA: HHI 2790, delta 682.
- Nashville MSA: HHI 2410, delta 476.
- Chattanooga MSA: HHI 2476, delta 600.
- Baton Rouge MSA: HHI 2582, delta 432.

The Greenville-Spartanburg MSA is the highest-risk geographic market based on the
structural presumption test under the 2023 Merger Guidelines (post-merger HHI > 1800
and delta > 100).
"""


def _write_corpus(tmp_root: Path) -> Path:
    repo_dir = tmp_root / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "market_concentration_memo.txt").write_text(
        REGULATORY_DOC, encoding="utf-8"
    )
    return repo_dir


@pytest.fixture
def matter():
    return MatterModel.open_in_memory()


def test_profile_dataset_shape_reads_documentcontent_and_registers_slots(matter):
    """The scout must extract text from DocumentContent and register slots."""
    from irys.core.repository import MatterRepository

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = _write_corpus(Path(tmp))
        repo = MatterRepository(repo_dir)

        engine = RLMEngine.__new__(RLMEngine)
        engine._matter_model = matter
        engine._emit_step = lambda *a, **kw: None  # no-op streaming

        # Build a minimal investigation state with a regulatory query
        state = InvestigationState.create(
            "antitrust HHI analysis for Meridian PeakAir acquisition",
            str(repo_dir),
        )

        all_files = repo.list_files()
        result = asyncio.run(engine._profile_dataset_shape(state, repo, all_files))

        assert result.get("enabled"), result
        # The corpus has 9 distinct MSAs; profiling should pick up most/all
        assert result["slots_registered"] >= 5, (
            f"expected to scout at least 5 MSA slots, got {result['slots_registered']}"
        )

        # Slots are persisted in the matter model
        opens = matter.extraction_slots.get_open_slots(
            matter.matter_id, slot_kind="collection_item",
        )
        assert len(opens) >= 5
        for slot in opens:
            assert slot["schema_ref"] == "legal.market_row.v1"
            assert slot["coverage_state"] == "pending"


def test_profile_skipped_for_non_regulatory_query(matter):
    """Non-regulatory queries must not trigger profiling."""
    from irys.core.repository import MatterRepository

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = _write_corpus(Path(tmp))
        repo = MatterRepository(repo_dir)
        engine = RLMEngine.__new__(RLMEngine)
        engine._matter_model = matter
        engine._emit_step = lambda *a, **kw: None

        # A plain contract review query should not match any regulatory term
        state = InvestigationState.create(
            "review the change of control provisions in this contract",
            str(repo_dir),
        )

        result = asyncio.run(
            engine._profile_dataset_shape(state, repo, repo.list_files())
        )
        assert result.get("enabled") is False
        assert matter.extraction_slots.get_open_slots(matter.matter_id) == []


def test_synthesis_row_summary_renders_when_slot_filled(matter):
    """After a market_row typed_evidence + slot fill, synthesis sees a table."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter

    # 1. Register a slot the way profile would
    sid, _ = matter.extraction_slots.register(
        matter.matter_id,
        "collection_item",
        "collection_item:legal.market_row.v1:msa:greenville-spartanburg",
        1,
        expected_count_confidence=0.85,
        scope_query_hash="abc",
        schema_ref="legal.market_row.v1",
    )

    # 2. Write a market_row typed_evidence record the way deep-read would
    rec_id, _ = matter.typed_evidence.upsert(
        "market_row",
        "market:greenville-spartanburg:test.pdf:row1",
        payload={
            "schema_ref": "legal.market_row.v1",
            "market_name": "Greenville-Spartanburg MSA",
            "product_market": "bulk industrial gas",
            "post_merger_hhi": 3224,
            "delta_hhi": 990,
            "structural_presumption": True,
            "risk_rating": "high",
            "source_detail": "p.7 / Concentration table",
            "source_document": "test.pdf",
        },
        document_id="test.pdf",
        confidence=0.9,
    )
    matter.extraction_slots.mark_filled(sid, rec_id)

    # 3. Build synthesis context for a regulatory query
    state = InvestigationState.create("antitrust HHI analysis", "/repo")
    summary = engine._build_extraction_slot_row_summary(state)
    assert summary, "synthesis row summary should be non-empty"
    assert "Greenville-Spartanburg MSA" in summary
    assert "3224" in summary
    assert "990" in summary
    assert "MARKET ROW SUMMARY" in summary


def test_synthesis_row_summary_empty_for_non_regulatory_query(matter):
    """Synthesis Context Principle: no slot table for non-regulatory queries."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter

    sid, _ = matter.extraction_slots.register(
        matter.matter_id,
        "collection_item",
        "collection_item:legal.market_row.v1:msa:greenville-spartanburg",
        1, expected_count_confidence=0.85, schema_ref="legal.market_row.v1",
    )
    rec_id, _ = matter.typed_evidence.upsert(
        "market_row",
        "market:greenville-spartanburg:test.pdf:row1",
        payload={"schema_ref": "legal.market_row.v1",
                 "market_name": "Greenville-Spartanburg MSA"},
        document_id="test.pdf", confidence=0.9,
    )
    matter.extraction_slots.mark_filled(sid, rec_id)

    state = InvestigationState.create("review change of control clause", "/repo")
    assert engine._build_extraction_slot_row_summary(state) == ""


def test_synthesis_row_summary_empty_when_no_slots(matter):
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter
    state = InvestigationState.create("antitrust HHI analysis", "/repo")
    assert engine._build_extraction_slot_row_summary(state) == ""


def test_termination_guard_blocks_when_high_confidence_slots_open(matter):
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter

    matter.extraction_slots.register(
        matter.matter_id, "collection_item",
        "collection_item:legal.market_row.v1:msa:atlanta", 1,
        expected_count_confidence=0.85, schema_ref="legal.market_row.v1",
    )
    state = InvestigationState.create("antitrust HHI analysis", "/repo")
    blocks, detail = engine._slot_coverage_blocks_termination(state)
    assert blocks
    assert "atlanta" in detail.lower() or "unfilled" in detail.lower()


def test_termination_guard_silent_for_non_regulatory_query(matter):
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter

    matter.extraction_slots.register(
        matter.matter_id, "collection_item", "k", 1,
        expected_count_confidence=0.85, schema_ref="legal.market_row.v1",
    )
    state = InvestigationState.create("review contract clause", "/repo")
    blocks, _ = engine._slot_coverage_blocks_termination(state)
    assert not blocks
