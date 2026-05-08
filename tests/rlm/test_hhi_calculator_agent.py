"""Tests for the HHI / Market-Share Calculator agent.

Operator Substrate Thesis: bounded operator, deterministic math, verifies
LLM-extracted values against computed truth.
"""

from __future__ import annotations

import asyncio

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation,
    AgentRequirement,
    AgentTaskView,
    HhiMarketShareCalculator,
    OperatorBudget,
    SubAgentDispatcher,
    SubAgentRegistry,
)
from irys.rlm.agents.hhi_calculator import _hhi_from_shares, _parse_share


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def test_parse_share_handles_percent_and_fraction():
    assert _parse_share("28%") == 0.28
    assert _parse_share("0.28") == 0.28
    assert _parse_share(28) == 0.28
    assert _parse_share(0.28) == 0.28
    assert _parse_share(None) is None
    assert _parse_share("not a number") is None
    assert _parse_share("") is None


def test_hhi_from_shares_pure_math():
    # Two firms at 50% each: HHI = 0.5^2 * 10000 + 0.5^2 * 10000 = 5000
    assert _hhi_from_shares([0.5, 0.5]) == 5000
    # Four firms at 25% each: HHI = 4 * 625 = 2500
    assert _hhi_from_shares([0.25] * 4) == 2500
    # 28% acquirer + 18% target + 22% C + 32% rest = pre-merger HHI
    pre = _hhi_from_shares([0.28, 0.18, 0.22, 0.32])
    assert pre == int(round((0.28**2 + 0.18**2 + 0.22**2 + 0.32**2) * 10000))


# ---------------------------------------------------------------------------
# Agent end-to-end (with stubbed matter model)
# ---------------------------------------------------------------------------


def _seed_market_row(matter: MatterModel, *, market_name: str, **kw):
    """Helper: write a market_row typed_evidence row matching the slot wedge schema."""
    payload = {"schema_ref": "legal.market_row.v1", "market_name": market_name, **kw}
    rec_id, _ = matter.typed_evidence.upsert(
        "market_row",
        f"market:{market_name.lower().replace(' ', '_')}:test.pdf:abc123",
        payload=payload,
        document_id="test.pdf",
        confidence=0.9,
    )
    return rec_id


def _make_invocation(matter_id: str, *, requirement=AgentRequirement.OPTIONAL):
    return AgentInvocation(
        matter_id=matter_id, run_id="r",
        agent_id="hhi", phase="pre_synthesis",
        persona_id=None, requirement=requirement,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="h",
    )


def test_hhi_agent_no_market_rows_returns_success_zero_artifacts():
    m = MatterModel.open_in_memory()
    agent = HhiMarketShareCalculator()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    assert len(out.invocations) == 1
    _, result = out.invocations[0]
    assert result.status == "success"
    assert result.artifacts == ()


def test_hhi_agent_computes_per_market_correctly():
    m = MatterModel.open_in_memory()
    _seed_market_row(
        m, market_name="Greenville-Spartanburg MSA",
        acquirer_share="28%", target_share="18%",
        other_shares=[
            {"name": "Competitor A", "share": "22%"},
            {"name": "Competitor B", "share": "32%"},
        ],
        # Extracted values match computed (no LLM hallucination here)
        pre_merger_hhi=2616, post_merger_hhi=3624, delta_hhi=1008,
    )
    agent = HhiMarketShareCalculator()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    assert len(result.artifacts) == 1
    art = result.artifacts[0]
    p = art.payload
    # 0.28² + 0.18² + 0.22² + 0.32² = 0.2616 → 2616
    assert p["computed_pre_hhi"] == 2616
    # Combined 0.46² + 0.22² + 0.32² = 0.3624 → 3624
    assert p["computed_post_hhi"] == 3624
    assert p["computed_delta_hhi"] == 1008
    assert p["structural_presumption"] is True
    assert p["share_sum_check"] == 1.0
    # Extracted values agreed → no discrepancies
    assert not p["discrepancies"]


def test_hhi_agent_flags_discrepancy_against_wrong_extraction():
    m = MatterModel.open_in_memory()
    _seed_market_row(
        m, market_name="Atlanta MSA",
        acquirer_share="30%", target_share="15%",
        other_shares=[{"name": "C1", "share": "25%"},
                      {"name": "C2", "share": "30%"}],
        # LLM hallucinated a wrong post-merger HHI
        pre_merger_hhi=2950, post_merger_hhi=4500, delta_hhi=900,
    )
    agent = HhiMarketShareCalculator()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    art = result.artifacts[0]
    p = art.payload
    # Real post HHI: 0.45^2 + 0.25^2 + 0.30^2 = 0.355 → 3550
    assert p["computed_post_hhi"] == 3550
    assert "post_merger_hhi" in p["discrepancies"]
    assert p["discrepancies"]["post_merger_hhi"]["extracted"] == 4500
    assert p["discrepancies"]["post_merger_hhi"]["computed"] == 3550
    assert art.verification_state == "candidate"


def test_hhi_agent_handles_share_sum_over_100_warning():
    m = MatterModel.open_in_memory()
    # Shares that sum > 1 (LLM error)
    _seed_market_row(
        m, market_name="Bogus Market",
        acquirer_share="40%", target_share="40%",
        other_shares=[{"name": "C1", "share": "50%"}],
    )
    agent = HhiMarketShareCalculator()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert any("share_sum_over_100" in w for w in result.warnings)


def test_hhi_agent_persists_artifacts_to_agent_artifact_table():
    m = MatterModel.open_in_memory()
    _seed_market_row(
        m, market_name="Market X",
        acquirer_share="0.30", target_share="0.20",
    )
    agent = HhiMarketShareCalculator()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    n = m.db.execute(
        "SELECT COUNT(*) FROM agent_artifact WHERE artifact_kind='hhi.calculation'"
    ).fetchone()[0]
    assert n == 1


def test_hhi_agent_capability_tags_include_hhi_and_compute():
    a = HhiMarketShareCalculator()
    assert "calculate.hhi" in a.capability_tags
    assert "compute.numerical" in a.capability_tags
    assert "verify.extraction" in a.capability_tags
