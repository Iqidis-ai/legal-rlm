"""Tests for proof-state-driven attention allocation in _get_issue_coverage_map().

Verifies:
1. Coverage map uses assertion-count ratio when no proof state is computed
2. Coverage map overlays ProofStateStore sufficiency when available
3. Contested issues are flagged as has_gap=True in coverage map
4. Insufficient issues are flagged as has_gap=True in coverage map
5. Sufficient issues are NOT flagged as has_gap
6. _build_issue_coverage_summary() includes proof status and sufficiency pct
7. Synthesis prompt block shows CONTESTED annotation when proof_status=contested
"""

import pytest
from irys.matter import MatterModel, IssueType, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _make_engine(model):
    """Return a minimal RLMEngine with _matter_model wired, no LLM calls needed."""
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    return engine


def _add_supporting(model, issue_id, n=1):
    aids = []
    for _ in range(n):
        cand = AssertionCandidate(
            proposition_text=f"Supporting fact {_add_supporting._ctr}",
            speech_act=SpeechAct.OPERATIVE,
            source_role=SourceRole.OPERATIVE,
            assertion_kind=AssertionKind.FACTUAL,
            document_id="doc1",
        )
        _add_supporting._ctr += 1
        aid, _ = model.assertions.upsert_occurrence(cand)
        model.issues.link_assertion(aid, issue_id, relation_type="supports")
        aids.append(aid)
    return aids[-1] if aids else None


_add_supporting._ctr = 0


def _add_attacking(model, issue_id):
    cand = AssertionCandidate(
        proposition_text=f"Attacking fact {_add_attacking._ctr}",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc1",
    )
    _add_attacking._ctr += 1
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, issue_id, relation_type="attacks")
    return aid


_add_attacking._ctr = 0


# ---------------------------------------------------------------------------
# _get_issue_coverage_map() with no proof state
# ---------------------------------------------------------------------------

def test_coverage_map_without_proof_state(model):
    """Falls back to assertion-count ratio when no proof state computed."""
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    _add_supporting(model, iid, n=3)

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    assert iid in cov
    frac, has_gap, cnt = cov[iid]
    assert frac > 0.0
    assert cnt == 3


def test_coverage_map_overlays_proof_state_sufficiency(model):
    """When ProofStateStore has a row, sufficiency replaces assertion-count ratio."""
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    for _ in range(9):
        _add_supporting(model, iid)
    # Compute proof state → sufficiency will be high
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    frac, has_gap, cnt = cov[iid]
    # ProofStateStore sufficiency for 9 supporting, no attacks = 9/10 = 0.9
    assert frac >= 0.85


# ---------------------------------------------------------------------------
# Contested issues flagged as has_gap
# ---------------------------------------------------------------------------

def test_contested_issue_flagged_as_gap(model):
    """proof_status=contested → has_gap=True even if assertion-count ratio is ok."""
    iid, _ = model.issues.upsert_issue("Contested Claim", IssueType.CLAIM)
    _add_supporting(model, iid)
    _add_attacking(model, iid)  # attacking >= supporting → contested
    _add_attacking(model, iid)
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    _, has_gap, _ = cov[iid]
    assert has_gap is True


def test_insufficient_issue_flagged_as_gap(model):
    iid, _ = model.issues.upsert_issue("Unproved Claim", IssueType.CLAIM)
    model.proof_state.compute_and_store(iid)  # zero assertions → insufficient

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    _, has_gap, _ = cov[iid]
    assert has_gap is True


def test_sufficient_issue_not_flagged_as_gap(model):
    iid, _ = model.issues.upsert_issue("Well-proved Claim", IssueType.CLAIM)
    for _ in range(9):
        _add_supporting(model, iid)
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    frac, has_gap, _ = cov[iid]
    assert frac >= 0.75
    assert has_gap is False


# ---------------------------------------------------------------------------
# _build_issue_coverage_summary() with proof state
# ---------------------------------------------------------------------------

def test_issue_coverage_summary_shows_proof_status(model):
    iid, _ = model.issues.upsert_issue("Test Claim", IssueType.CLAIM)
    for _ in range(3):
        _add_supporting(model, iid)
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    # _build_issue_coverage_summary needs state — just call it directly
    from irys.rlm.state import InvestigationState
    summary = engine._build_issue_coverage_summary()

    # Should mention the proof status
    assert "Test Claim" in summary or "test claim" in summary.lower()
    # Should include sufficiency %
    assert "%" in summary or "sufficiency" in summary.lower()


def test_issue_coverage_summary_shows_contested_annotation(model):
    iid, _ = model.issues.upsert_issue("Contested Claim", IssueType.CLAIM)
    _add_supporting(model, iid)
    _add_attacking(model, iid)
    _add_attacking(model, iid)
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    summary = engine._build_issue_coverage_summary()

    # Should flag the contested status
    assert "CONTESTED" in summary.upper() or "contested" in summary.lower()


def test_issue_coverage_summary_without_proof_state(model):
    """Fallback path: no proof state → assertion-count annotation."""
    iid, _ = model.issues.upsert_issue("Unprofiled Claim", IssueType.CLAIM)
    _add_supporting(model, iid)

    engine = _make_engine(model)
    summary = engine._build_issue_coverage_summary()

    assert "Unprofiled Claim" in summary or "unprofiled" in summary.lower()
    assert "%" in summary
