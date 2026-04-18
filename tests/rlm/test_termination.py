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
    # Lead has no lead_ev_floor on the dataclass by default — we're
    # testing the fallthrough path; the engine helper reads
    # getattr(contract, "lead_ev_floor", default).
    leads = [_lead(0.3), _lead(0.6), _lead(0.75), _lead(0.9)]
    viable = engine._viable_leads(leads, contract=contract)
    assert len(viable) == 2
    assert all(l.priority >= 0.7 for l in viable)


# ---------------------------------------------------------------------------
# Integration: the new controller respects contract.min_iter
# ---------------------------------------------------------------------------


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
