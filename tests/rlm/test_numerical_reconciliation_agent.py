"""Tests for the Numerical Reconciliation Agent (QoE / EBITDA / NWC / PPA)."""

from __future__ import annotations

import asyncio

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation, AgentRequirement, AgentTaskView,
    NumericalReconciliationAgent, OperatorBudget,
    SubAgentDispatcher, SubAgentRegistry,
)
from irys.rlm.agents.numerical_reconciliation import _parse_money, _format_money


# ---------------------------------------------------------------------------
# Money parsing
# ---------------------------------------------------------------------------


def test_parse_money_handles_dollar_million():
    assert _parse_money("$0.8M") == 800_000.0
    assert _parse_money("$1.3M") == 1_300_000.0


def test_parse_money_handles_parentheses_negation():
    assert _parse_money("(1.3M)") == -1_300_000.0
    assert _parse_money("($500K)") == -500_000.0


def test_parse_money_handles_commas_and_thousands():
    assert _parse_money("$1,234,567") == 1_234_567.0
    assert _parse_money("500K") == 500_000.0
    assert _parse_money("3 billion") == 3_000_000_000.0


def test_parse_money_returns_none_for_junk():
    for s in ["", "—", "N/A", "n/a", "TBD", None, "no number here"]:
        assert _parse_money(s) is None


def test_parse_money_passthrough_numeric():
    assert _parse_money(42) == 42.0
    assert _parse_money(3.14) == 3.14


def test_format_money():
    assert _format_money(1_500_000) == "$1.50M"
    assert _format_money(2_500) == "$2.5K"
    assert _format_money(None) == "—"


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------


def _seed_qoe(matter: MatterModel, **kw):
    payload = {"schema_ref": "legal.qoe_line_item.v1", **kw}
    label = kw.get("line_item_label", "x")
    cat = kw.get("category", "ebitda_bridge")
    period = kw.get("period", "fy2024")
    rec_id, _ = matter.typed_evidence.upsert(
        "qoe_line_item",
        f"qoe:{cat}:{period}:{label}:doc.pdf:{abs(hash(label)) & 0xFFFF:04x}",
        payload=payload,
        document_id="doc.pdf", confidence=0.9,
    )
    return rec_id


def _make_invocation(matter_id: str):
    return AgentInvocation(
        matter_id=matter_id, run_id="r",
        agent_id="recon", phase="pre_synthesis",
        persona_id=None, requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="h",
    )


def test_recon_no_qoe_returns_success_zero_artifacts():
    m = MatterModel.open_in_memory()
    agent = NumericalReconciliationAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    assert result.artifacts == ()


def test_recon_groups_by_category_period_and_computes_totals():
    m = MatterModel.open_in_memory()
    # Two-line EBITDA bridge for FY2024
    _seed_qoe(m, category="EBITDA bridge", period="FY2024",
              schedule="EBITDA Bridge", currency="USD",
              line_item_label="Owner comp normalization",
              seller_value="$0.8M", buyer_value="$1.3M", delta="-$0.5M")
    _seed_qoe(m, category="EBITDA bridge", period="FY2024",
              schedule="EBITDA Bridge", currency="USD",
              line_item_label="Legal settlement",
              seller_value="$1.0M", buyer_value="$1.8M", delta="-$0.8M")
    # Different period — separate group
    _seed_qoe(m, category="EBITDA bridge", period="FY2023",
              schedule="EBITDA Bridge", currency="USD",
              line_item_label="Inventory adjustment",
              seller_value="$0.4M", buyer_value="$0.0M", delta="$0.4M")

    agent = NumericalReconciliationAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    # Two artifacts (one per period)
    assert len(result.artifacts) == 2
    fy2024 = [a for a in result.artifacts if a.payload.get("period") == "fy2024"][0]
    p = fy2024.payload
    assert p["n_line_items"] == 2
    assert p["computed_seller_total"] == 1_800_000.0
    assert p["computed_buyer_total"] == 3_100_000.0
    # delta_total = -0.5M + -0.8M = -1.3M
    assert p["computed_delta_total"] == -1_300_000.0
    # implied_delta = 3.1M - 1.8M = 1.3M; sum_delta = -1.3M (signs differ
    # because bridge points the other way) — still consistent in magnitude
    # but our agent compares signs, so this is by design "inconsistent".
    # That's correct — the LLM emitted negative deltas while the implied
    # bridge is positive, signaling the LLM's sign convention is off.
    assert p["bridge_consistent"] is False


def test_recon_bridge_consistent_when_signs_align():
    m = MatterModel.open_in_memory()
    _seed_qoe(m, category="NWC", period="LTM Mar 2025",
              schedule="NWC schedule", currency="USD",
              line_item_label="AR adjustment",
              seller_value="$1.0M", buyer_value="$1.5M", delta="$0.5M")
    _seed_qoe(m, category="NWC", period="LTM Mar 2025",
              schedule="NWC schedule", currency="USD",
              line_item_label="Inventory adjustment",
              seller_value="$2.0M", buyer_value="$1.5M", delta="-$0.5M")
    agent = NumericalReconciliationAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    art = result.artifacts[0]
    p = art.payload
    # seller=3.0M, buyer=3.0M → implied delta = 0.0M
    # sum of deltas = 0.5M + (-0.5M) = 0.0M → consistent
    assert p["computed_seller_total"] == 3_000_000.0
    assert p["computed_buyer_total"] == 3_000_000.0
    assert p["computed_delta_total"] == 0.0
    assert p["bridge_consistent"] is True
    assert art.verification_state == "verified"


def test_recon_persists_artifacts():
    m = MatterModel.open_in_memory()
    _seed_qoe(m, category="EBITDA", period="FY2024",
              line_item_label="x", seller_value="$1M", buyer_value="$2M",
              delta="$1M")
    agent = NumericalReconciliationAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    n = m.db.execute(
        "SELECT COUNT(*) FROM agent_artifact WHERE artifact_kind='numeric.reconciliation'"
    ).fetchone()[0]
    assert n == 1


def test_recon_capability_tags():
    a = NumericalReconciliationAgent()
    assert "compute.numerical" in a.capability_tags
    assert "calculate.financial" in a.capability_tags
    assert "verify.extraction" in a.capability_tags
