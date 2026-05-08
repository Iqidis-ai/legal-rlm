"""Integration test: engine pre-synthesis operator hook fires and produces
agent_artifact rows that flow into the synthesis context.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from irys.matter import MatterModel
from irys.rlm.engine import RLMEngine
from irys.rlm.state import InvestigationState


def _make_engine_with_matter() -> tuple[RLMEngine, MatterModel]:
    matter = MatterModel.open_in_memory()
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = matter
    return engine, matter


def _seed_market_row(matter: MatterModel, *, market_name: str, **kw):
    payload = {"schema_ref": "legal.market_row.v1", "market_name": market_name, **kw}
    rec_id, _ = matter.typed_evidence.upsert(
        "market_row",
        f"market:{market_name.lower().replace(' ', '_')}",
        payload=payload, document_id="test.pdf", confidence=0.9,
    )
    return rec_id


def test_pre_synthesis_operator_hook_runs_hhi_calculator():
    """Seed a market_row → run hook → HHI artifact appears in agent_artifact."""
    engine, matter = _make_engine_with_matter()
    _seed_market_row(
        matter, market_name="Greenville-Spartanburg MSA",
        acquirer_share="28%", target_share="18%",
        other_shares=[{"name": "C1", "share": "22%"},
                      {"name": "C2", "share": "32%"}],
        pre_merger_hhi=2616, post_merger_hhi=3624, delta_hhi=1008,
    )

    state = InvestigationState.create(
        "antitrust analysis with HHI", "/tmp/repo",
    )
    asyncio.run(engine._run_pre_synthesis_operators(state))

    # Artifact persisted to agent_artifact
    rows = matter.db.execute(
        "SELECT artifact_kind, label, payload_json FROM agent_artifact"
    ).fetchall()
    kinds = [r["artifact_kind"] for r in rows]
    assert "hhi.calculation" in kinds

    # Operator phase summary recorded in findings
    summary = state.findings.get("operator_phase_summary")
    assert summary is not None
    assert summary["n_artifacts"] >= 1


def test_build_agent_artifact_summary_renders_hhi_table():
    engine, matter = _make_engine_with_matter()
    _seed_market_row(
        matter, market_name="Atlanta MSA",
        acquirer_share="30%", target_share="15%",
        other_shares=[{"name": "C1", "share": "25%"}],
    )
    state = InvestigationState.create("HHI", "/tmp/repo")
    asyncio.run(engine._run_pre_synthesis_operators(state))

    summary = engine._build_agent_artifact_summary(state)
    assert summary, "expected non-empty operator artifact summary"
    assert "OPERATOR-COMPUTED HHI" in summary
    assert "Atlanta MSA" in summary


def test_pre_synthesis_operator_hook_is_silent_when_no_evidence():
    """Operator hook should not crash when there's no qualifying evidence."""
    engine, matter = _make_engine_with_matter()
    state = InvestigationState.create("anything", "/tmp/repo")
    asyncio.run(engine._run_pre_synthesis_operators(state))
    # No artifacts, but the hook completed. Operator phase summary should
    # exist with zero artifacts.
    summary = state.findings.get("operator_phase_summary")
    assert summary is not None
    assert summary["n_artifacts"] == 0


def test_build_agent_artifact_summary_empty_when_no_artifacts():
    engine, matter = _make_engine_with_matter()
    state = InvestigationState.create("query", "/tmp/repo")
    assert engine._build_agent_artifact_summary(state) == ""


def test_qoe_reconciliation_flows_through_engine():
    """Seed qoe_line_item → engine hook → numeric.reconciliation artifact."""
    engine, matter = _make_engine_with_matter()
    payload_a = {
        "schema_ref": "legal.qoe_line_item.v1",
        "category": "EBITDA bridge", "period": "FY2024",
        "schedule": "EBITDA Bridge", "currency": "USD",
        "line_item_label": "Owner comp",
        "seller_value": "$0.8M", "buyer_value": "$1.3M", "delta": "$0.5M",
    }
    matter.typed_evidence.upsert(
        "qoe_line_item", "qoe:ebitda:FY2024:owner_comp:doc.pdf:abc",
        payload=payload_a, document_id="doc.pdf", confidence=0.9,
    )

    state = InvestigationState.create("analyze qoe reconciliation", "/tmp/repo")
    asyncio.run(engine._run_pre_synthesis_operators(state))

    rows = matter.db.execute(
        "SELECT artifact_kind FROM agent_artifact"
    ).fetchall()
    kinds = [r["artifact_kind"] for r in rows]
    assert "numeric.reconciliation" in kinds

    summary = engine._build_agent_artifact_summary(state)
    assert "OPERATOR-COMPUTED RECONCILIATION" in summary
