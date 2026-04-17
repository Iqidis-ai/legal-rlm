"""Tests for proof-state-enriched attention allocation in _get_issue_coverage_map().

Verifies:
1. Coverage map reports nonzero coverage_fraction from get_issue_coverage_report()
   when no proof_state row exists.
2. Coverage map preserves the report's weighted coverage_fraction even when a
   proof_state row has a divergent raw sufficiency score (PR.2 contract:
   report is canonical for coverage_fraction; proof_state only enriches
   has_proof_gap).
3. Contested issues (proof_status=contested on proof_state) are elevated to
   has_gap=True in the map.
4. Insufficient issues are elevated to has_gap=True in the map.
5. Sufficient issues are NOT flagged as has_gap.
6. _build_issue_coverage_summary() includes proof status and sufficiency pct.
7. Synthesis prompt block shows CONTESTED annotation when proof_status=contested.
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


def test_coverage_map_preserves_report_fraction_when_proof_state_exists(model):
    """PR.2: the weighted coverage_fraction from get_issue_coverage_report() is
    canonical. proof_state's raw-count sufficiency must NOT replace it — only
    the has_proof_gap flag is enriched from proof_state. Two default
    record_fact-style supports with no predicates produce a known divergence:
    the report's weighted fallback is 2/3 = 0.667 while proof_state's raw
    sufficiency is 2/3 ≈ 0.667 as well, so we use a materiality/attacker shape
    where the two diverge — one supporting, one attacking produces
    proof_state.proof_status=contested with sufficiency=0 but the report still
    shows weighted coverage > 0."""
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    _add_supporting(model, iid)
    _add_attacking(model, iid)
    model.proof_state.compute_and_store(iid)

    # Confirm the fixture actually produces divergent report vs proof_state:
    # report.coverage_fraction > 0 from weighted support, proof_state is contested.
    report_row = next(r for r in model.get_issue_coverage_report() if r["id"] == iid)
    assert report_row["coverage_fraction"] > 0, (
        "fixture precondition — report must compute nonzero weighted coverage"
    )

    engine = _make_engine(model)
    cov = engine._get_issue_coverage_map()

    frac, has_gap, cnt = cov[iid]
    # Map must preserve the report's weighted coverage_fraction unchanged.
    assert frac == pytest.approx(report_row["coverage_fraction"]), (
        f"Coverage map must preserve report fraction {report_row['coverage_fraction']}, "
        f"got {frac}"
    )
    # proof_state.proof_status='contested' must elevate has_gap.
    assert has_gap is True


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


def test_issue_coverage_summary_shows_advocacy_only_annotation(model):
    """Issues backed only by advocacy sources get ⚠ ADVOCACY-ONLY annotation (SO-5)."""
    iid, _ = model.issues.upsert_issue("Advocacy Claim", IssueType.CLAIM)
    # Add only advocacy assertions
    for i in range(2):
        cand = AssertionCandidate(
            proposition_text=f"Advocacy fact {i}",
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.ADVOCACY,
            assertion_kind=AssertionKind.FACTUAL,
            document_id="complaint.pdf",
        )
        aid, _ = model.assertions.upsert_occurrence(cand)
        model.issues.link_assertion(aid, iid, relation_type="supports")
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    summary = engine._build_issue_coverage_summary()

    assert "ADVOCACY-ONLY" in summary


def test_coverage_map_flags_advocacy_only_as_has_gap(model):
    """advocacy_only=True issues get has_gap=True in _get_issue_coverage_map() (SO-5)."""
    iid, _ = model.issues.upsert_issue("Advocacy Issue", IssueType.CLAIM)
    cand = AssertionCandidate(
        proposition_text="Advocacy only fact",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="brief.pdf",
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, iid, relation_type="supports")
    model.proof_state.compute_and_store(iid)

    engine = _make_engine(model)
    coverage_map = engine._get_issue_coverage_map()

    # Issue should be flagged as having a gap because evidence is advocacy-only
    assert iid in coverage_map
    _coverage, has_gap, _cnt = coverage_map[iid]
    assert has_gap is True
