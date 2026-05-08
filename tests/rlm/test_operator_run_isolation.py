"""Regression tests for Codex PR-gate round-2 fixes (HOLD-1, HOLD-2,
work-profile schema/scoping).

Covers:
  - Cross-run artifact exclusion: a second run on the same matter must
    NOT see the first run's agent_artifact in its synthesis context.
  - synthesis_input_artifact rows carry the correct (non-blank) run_id.
  - _compute_agent_work_profile returns nonzero counts after seeded
    typed_evidence_record rows (catches the typed_evidence vs
    typed_evidence_record table-name regression).
  - Work profile is matter-scoped (no cross-matter bleed).
"""

from __future__ import annotations

import asyncio

from irys.matter import MatterModel
from irys.rlm.engine import RLMEngine
from irys.rlm.state import InvestigationState


def _make_engine(matter: MatterModel) -> RLMEngine:
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter
    return engine


def _seed_market_row(matter: MatterModel, market_name: str) -> None:
    payload = {
        "schema_ref": "legal.market_row.v1",
        "market_name": market_name,
        "acquirer_share": "30%",
        "target_share": "20%",
        "other_shares": [{"name": "C1", "share": "25%"},
                         {"name": "C2", "share": "25%"}],
    }
    matter.typed_evidence.upsert(
        "market_row",
        f"market:{market_name.lower().replace(' ', '_')}",
        payload=payload, document_id="test.pdf", confidence=0.9,
    )


# ---------------------------------------------------------------------------
# work_profile correctness
# ---------------------------------------------------------------------------


def test_work_profile_counts_typed_evidence_record_market_row():
    """After seeding a market_row, work_profile must report >= 1.

    Catches the round-2 regression where engine queried the wrong table
    (`typed_evidence` instead of `typed_evidence_record`) and silently
    returned 0.
    """
    matter = MatterModel.open_in_memory()
    engine = _make_engine(matter)

    # Empty matter — counts should be zero, not error.
    profile_empty = engine._compute_agent_work_profile()
    assert profile_empty["market_row_count"] == 0
    assert profile_empty["qoe_line_item_count"] == 0
    assert profile_empty["cp_gap_count"] == 0

    _seed_market_row(matter, "Boston MSA")

    profile = engine._compute_agent_work_profile()
    assert profile["market_row_count"] >= 1, (
        f"work_profile market_row_count should be >= 1 after seed; got "
        f"{profile['market_row_count']}. This regression caught the "
        f"typed_evidence vs typed_evidence_record table-name bug."
    )


def test_work_profile_is_matter_scoped():
    """Two matters in separate DBs — each profile reports only its own."""
    matter_a = MatterModel.open_in_memory()
    matter_b = MatterModel.open_in_memory()
    _seed_market_row(matter_a, "Atlanta MSA")
    # matter_b stays empty

    engine_a = _make_engine(matter_a)
    engine_b = _make_engine(matter_b)

    assert engine_a._compute_agent_work_profile()["market_row_count"] >= 1
    assert engine_b._compute_agent_work_profile()["market_row_count"] == 0


# ---------------------------------------------------------------------------
# cross-run artifact exclusion (HOLD-1)
# ---------------------------------------------------------------------------


def test_build_agent_artifact_summary_excludes_other_run_artifacts():
    """Run-1 produces an HHI artifact; run-2 starts fresh on same matter.
    Run-2's synthesis context must NOT include run-1 artifacts.
    """
    matter = MatterModel.open_in_memory()
    engine = _make_engine(matter)

    _seed_market_row(matter, "Run1 Market")
    state1 = InvestigationState.create("HHI run 1", "/tmp/repo")
    state1._run_id = "run-1"
    asyncio.run(engine._run_pre_synthesis_operators(state1))

    # Sanity: run-1 produced an HHI artifact.
    s1 = engine._build_agent_artifact_summary(state1)
    assert "Run1 Market" in s1

    # Now start "run-2" — DO NOT seed any new market_row, DO NOT call the
    # operator hook. The run-2 summary should be empty (no artifacts
    # belong to run-2).
    state2 = InvestigationState.create("HHI run 2", "/tmp/repo")
    state2._run_id = "run-2"
    s2 = engine._build_agent_artifact_summary(state2)
    assert s2 == "", (
        f"run-2 summary leaked run-1 artifacts. Got: {s2!r}. "
        f"This regression catches HOLD-1 cross-run bleed."
    )


# ---------------------------------------------------------------------------
# synthesis_input_artifact run_id correctness (HOLD-2)
# ---------------------------------------------------------------------------


def test_synthesis_input_artifact_carries_correct_run_id():
    """synthesis_input_artifact rows must record the actual run_id, not blank."""
    matter = MatterModel.open_in_memory()
    engine = _make_engine(matter)

    _seed_market_row(matter, "Audit Trail Market")
    state = InvestigationState.create("HHI", "/tmp/repo")
    state._run_id = "real-run-abc"
    asyncio.run(engine._run_pre_synthesis_operators(state))

    # Build summary so synthesis_input_artifact rows are recorded.
    summary = engine._build_agent_artifact_summary(state)
    assert summary, "expected non-empty summary"

    rows = matter.db.execute(
        "SELECT run_id FROM synthesis_input_artifact"
    ).fetchall()
    assert rows, "expected at least one synthesis_input_artifact row"
    run_ids = {r["run_id"] for r in rows}
    assert run_ids == {"real-run-abc"}, (
        f"synthesis_input_artifact run_id must be 'real-run-abc'; "
        f"got {run_ids}. Catches HOLD-2 blank-run-id audit bug."
    )


def test_sub_agent_invocation_carries_correct_run_id():
    """sub_agent_invocation rows must record the actual run_id, not blank."""
    matter = MatterModel.open_in_memory()
    engine = _make_engine(matter)

    _seed_market_row(matter, "Invocation Trail Market")
    state = InvestigationState.create("HHI", "/tmp/repo")
    state._run_id = "real-run-xyz"
    asyncio.run(engine._run_pre_synthesis_operators(state))

    rows = matter.db.execute(
        "SELECT run_id FROM sub_agent_invocation"
    ).fetchall()
    assert rows, "expected at least one sub_agent_invocation row"
    run_ids = {(r["run_id"] or "") for r in rows}
    assert "real-run-xyz" in run_ids, (
        f"sub_agent_invocation run_id must include 'real-run-xyz'; "
        f"got {run_ids}."
    )
    assert "" not in run_ids, (
        f"sub_agent_invocation must not record blank run_id; got {run_ids}."
    )
