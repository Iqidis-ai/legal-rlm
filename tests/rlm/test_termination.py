"""MVI-3 cold-loop termination rewrite tests.

Covers the new governed-progress helpers on RLMEngine:
  - _no_material_answerability_delta
  - _dead_loop_detected
  - _viable_leads (contract-aware floor)
  - _target_is_sufficient fallback to citation floor when no issues

These are pure-function tests against a minimally constructed
InvestigationState — no Gemini API, no repo. The loop integration
is covered separately by the full-suite regression.
"""

from __future__ import annotations

import pytest

from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.governance import CascadeGovernor, ExecutionContract
from irys.rlm.state import InvestigationState, Lead


class _StubClient:
    async def complete(self, *a, **kw):  # pragma: no cover — not called in these tests
        raise AssertionError("LLM should not be called from pure-function checks")


@pytest.fixture
def engine():
    # Matter model stays None — pure-function path.
    return RLMEngine(gemini_client=_StubClient(), config=RLMConfig())


def _state(**overrides) -> InvestigationState:
    s = InvestigationState.create(
        "test query", "/tmp/repo",
        research_mode="deep",
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# _no_material_answerability_delta
# ---------------------------------------------------------------------------


def test_no_delta_when_coverage_flat(engine):
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.2, 0.2, 0.2],
        open_gap_count_per_iteration=[3, 3, 3, 3],
    )
    assert engine._no_material_answerability_delta(s) is True


def test_material_delta_when_coverage_moves(engine):
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.2, 0.3, 0.5],
        open_gap_count_per_iteration=[3, 3, 3, 3],
    )
    assert engine._no_material_answerability_delta(s) is False


def test_material_delta_when_gap_closes(engine):
    s = _state(
        coverage_sum_per_iteration=[0.2, 0.2, 0.2, 0.2],
        open_gap_count_per_iteration=[3, 3, 3, 2],  # gap closed in last iter
    )
    assert engine._no_material_answerability_delta(s) is False


def test_no_delta_insufficient_history(engine):
    """Need at least 3 samples — with fewer, we can't judge delta."""
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.2],
        open_gap_count_per_iteration=[3, 3],
    )
    assert engine._no_material_answerability_delta(s) is False


# ---------------------------------------------------------------------------
# _dead_loop_detected
# ---------------------------------------------------------------------------


def test_dead_loop_three_iterations_no_progress(engine):
    """3 iterations all flat on coverage AND no gap close."""
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.2, 0.2, 0.2, 0.2],
        open_gap_count_per_iteration=[3, 3, 3, 3, 3],
    )
    assert engine._dead_loop_detected(s) is True


def test_dead_loop_not_detected_when_gap_closes_once(engine):
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.2, 0.2, 0.2, 0.2],
        open_gap_count_per_iteration=[3, 3, 2, 2, 2],  # gap closed mid-run
    )
    assert engine._dead_loop_detected(s) is False


def test_dead_loop_insufficient_history(engine):
    s = _state(
        coverage_sum_per_iteration=[0.0, 0.0, 0.0],
        open_gap_count_per_iteration=[3, 3, 3],
    )
    assert engine._dead_loop_detected(s) is False


# ---------------------------------------------------------------------------
# _viable_leads
# ---------------------------------------------------------------------------


def _lead(priority: float) -> Lead:
    return Lead.create(
        description="test lead",
        priority=priority,
        source="test",
    )


def test_viable_leads_default_floor(engine):
    leads = [_lead(0.1), _lead(0.4), _lead(0.6), _lead(0.9)]
    # Default floor is 0.5 when contract is None.
    viable = engine._viable_leads(leads, contract=None)
    assert len(viable) == 2
    assert all(l.priority >= 0.5 for l in viable)


def test_viable_leads_contract_floor(engine):
    contract = ExecutionContract(family="investigate", lead_ev_floor=0.7)
    leads = [_lead(0.3), _lead(0.6), _lead(0.75), _lead(0.9)]
    viable = engine._viable_leads(leads, contract=contract)
    assert len(viable) == 2
    assert all(l.priority >= 0.7 for l in viable)


def test_lead_ev_floor_zero_stays_zero(engine):
    """Adversarial #10 Fix E: `lead_ev_floor=0.0` must mean 'no floor'
    — every lead passes. Old `or floor` idiom silently coerced 0.0
    back to 0.5, giving partial gating when the caller explicitly
    turned it off."""
    contract = ExecutionContract(family="investigate", lead_ev_floor=0.0)
    leads = [_lead(0.001), _lead(0.05), _lead(0.5), _lead(0.9)]
    viable = engine._viable_leads(leads, contract=contract)
    # floor=0.0 → every lead with priority >= 0.0 viable (all).
    assert len(viable) == 4


def test_lead_ev_floor_nan_and_inf_rejected(engine, caplog):
    """NaN and ±inf must be rejected and default to 0.5 (logged),
    not silently let everything through or filter everything out."""
    import math
    for bad in (float("nan"), float("inf"), float("-inf")):
        contract = ExecutionContract(family="investigate", lead_ev_floor=bad)
        leads = [_lead(0.1), _lead(0.4), _lead(0.6), _lead(0.9)]
        viable = engine._viable_leads(leads, contract=contract)
        # With fallback floor=0.5, only priorities >= 0.5 are viable.
        assert len(viable) == 2, f"bad floor {bad} did not default to 0.5"


def test_lead_ev_floor_non_numeric_rejected(engine):
    """A string or None must not crash or behave unpredictably —
    coerce or default to 0.5."""
    contract = ExecutionContract(family="investigate", lead_ev_floor="not a number")
    leads = [_lead(0.1), _lead(0.6)]
    viable = engine._viable_leads(leads, contract=contract)
    # Defaulted to 0.5 → priority 0.6 passes, 0.1 doesn't.
    assert len(viable) == 1


def test_viable_leads_ev_mode(engine):
    """When leads carry expected_cost + expected_coverage_gain, the
    contract's lead_ev_floor is interpreted as coverage-per-dollar."""
    def _ev_lead(cost: float, gain: float) -> Lead:
        return Lead.create(
            description="ev test",
            source="test",
            priority=0.1,  # low — EV path should override
            expected_cost_usd=cost,
            expected_coverage_gain=gain,
        )
    # ev_score values: 10, 50, 100, 5
    leads = [
        _ev_lead(0.01, 0.10),   # ev=10
        _ev_lead(0.002, 0.10),  # ev=50
        _ev_lead(0.001, 0.10),  # ev=100
        _ev_lead(0.02, 0.10),   # ev=5
    ]
    # Floor=20 → only the two high-EV leads pass.
    contract = ExecutionContract(family="investigate", lead_ev_floor=20.0)
    viable = engine._viable_leads(leads, contract=contract)
    assert len(viable) == 2
    assert all(l.ev_score >= 20.0 for l in viable)


# ---------------------------------------------------------------------------
# Integration: the new controller respects contract.min_iter
# ---------------------------------------------------------------------------


def test_pre_spend_ev_gate_filters_low_ev_high_priority_leads(engine):
    """Adv#11 Fix 2 regression: a lead with high priority (passes
    min_lead_priority) but below-floor EV must be filtered BEFORE task
    dispatch. Previously _viable_leads only ran inside
    _should_continue_investigation — termination — which means the
    low-EV batch dispatched once before the loop exited.

    Reproduces the Codex repro: priority=0.9, cost=1.0, gain=0.1
    (ev=0.1) against lead_ev_floor=0.5 must return an empty viable
    list even though the lead clears the legacy priority threshold.
    """
    lead = Lead.create(
        description="low EV but high priority",
        source="test",
        priority=0.9,
    )
    lead.expected_cost_usd = 1.0
    lead.expected_coverage_gain = 0.1
    contract = ExecutionContract(family="investigate", lead_ev_floor=0.5)
    # EV = 0.1 < floor 0.5 → not viable, even though priority >= 0.3.
    assert lead.ev_score == 0.1
    assert lead.priority >= engine.config.min_lead_priority
    viable = engine._viable_leads([lead], contract=contract)
    assert viable == []


def test_min_iter_from_contract_gates_termination(engine):
    """Contract min_iter=0 allows termination at iteration 1 if target
    is sufficient; min_iter=2 forces a second iteration."""
    s = _state(
        max_depth_reached=1,
        coverage_sum_per_iteration=[0.0, 0.0],
        open_gap_count_per_iteration=[0, 0],
    )
    s.execution_contract = ExecutionContract(family="read", min_iter=0)
    # Citation floor gate — no citations yet, so even with min_iter=0
    # we shouldn't declare sufficient but also shouldn't be held by
    # min_iter.
    # Add a citation so the citation-floor path can fire.
    s.add_citation(
        document="doc.pdf", page=None, text="test",
        context="test", relevance="supporting",
    )
    should_continue, _reason = engine._should_continue_investigation(s)
    # With min_iter=0, citation floor met, no pending leads → should stop.
    assert should_continue is False


def test_min_iter_default_floor(engine):
    """Without a contract, fall back to research_profile.min_depth."""
    s = _state(
        max_depth_reached=0,  # haven't reached min_depth yet
        coverage_sum_per_iteration=[],
        open_gap_count_per_iteration=[],
    )
    should_continue, reason = engine._should_continue_investigation(s)
    # min_depth for "deep" mode is > 0, so we should be forced to continue.
    assert should_continue is True
    assert "minimum evidence" in reason.lower()
