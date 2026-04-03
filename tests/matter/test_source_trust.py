"""Tests for SO-5 source role trust enforcement in ProofStateStore.

Verifies that source role is an operational enforcement signal, not just metadata:
1.  trust_weighted_support present in compute_and_store() result
2.  trust_weighted_support present in get() result
3.  trust_weighted_support present in get_all() results
4.  Operative assertion has trust weight 1.0
5.  Advocacy assertion has trust weight 0.3
6.  Authoritative assertion has trust weight 1.0
7.  advocacy_only=False when supporting assertions include operative sources
8.  advocacy_only=True when all supporting assertions are advocacy
9.  advocacy_only=False when no supporting assertions exist
10. Mixed operative+advocacy gives trust_weighted_support between counts
11. trust_weighted_attack populated for attacking assertions
12. get_gaps() results include trust fields
13. get_by_status() results include trust fields
14. Unknown source_role defaults to 0.5 trust
15. Recompute updates advocacy_only correctly
"""

import pytest
from irys.matter import MatterModel, IssueType, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate
from irys.matter.graph import ProofStateStore


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def issue_id(model):
    iid, _ = model.issues.upsert_issue("Liability", IssueType.CLAIM)
    return iid


def _add(model, issue_id, source_role, relation="supports"):
    static = getattr(_add, "_ctr", 0)
    _add._ctr = static + 1
    cand = AssertionCandidate(
        proposition_text=f"Fact {static}",
        speech_act=SpeechAct.ALLEGED,
        source_role=source_role,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, issue_id, relation_type=relation)
    return aid


# ---------------------------------------------------------------------------
# 1-3. trust fields present in returned dicts
# ---------------------------------------------------------------------------

def test_trust_fields_in_compute_result(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)
    result = model.proof_state.compute_and_store(issue_id)
    assert "trust_weighted_support" in result
    assert "trust_weighted_attack" in result
    assert "advocacy_only" in result


def test_trust_fields_in_get(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)
    model.proof_state.compute_and_store(issue_id)
    ps = model.proof_state.get(issue_id)
    assert ps is not None
    assert "trust_weighted_support" in ps
    assert "advocacy_only" in ps


def test_trust_fields_in_get_all(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)
    model.proof_state.compute_and_store(issue_id)
    all_ps = model.proof_state.get_all()
    assert len(all_ps) == 1
    assert "trust_weighted_support" in all_ps[0]


# ---------------------------------------------------------------------------
# 4-6. Source role trust weights
# ---------------------------------------------------------------------------

def test_operative_weight_is_1_0(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["trust_weighted_support"] == pytest.approx(
        ProofStateStore.SOURCE_TRUST["operative"], abs=0.01
    )


def test_advocacy_weight_is_0_3(model, issue_id):
    _add(model, issue_id, SourceRole.ADVOCACY)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["trust_weighted_support"] == pytest.approx(
        ProofStateStore.SOURCE_TRUST["advocacy"], abs=0.01
    )


def test_authoritative_weight_is_1_0(model, issue_id):
    _add(model, issue_id, SourceRole.AUTHORITATIVE)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["trust_weighted_support"] == pytest.approx(
        ProofStateStore.SOURCE_TRUST["authoritative"], abs=0.01
    )


# ---------------------------------------------------------------------------
# 7-9. advocacy_only flag
# ---------------------------------------------------------------------------

def test_advocacy_only_false_when_operative_present(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)
    _add(model, issue_id, SourceRole.ADVOCACY)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["advocacy_only"] is False


def test_advocacy_only_true_when_all_advocacy(model, issue_id):
    _add(model, issue_id, SourceRole.ADVOCACY)
    _add(model, issue_id, SourceRole.ADVOCACY)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["advocacy_only"] is True


def test_advocacy_only_false_when_no_supporting(model, issue_id):
    result = model.proof_state.compute_and_store(issue_id)
    assert result["advocacy_only"] is False


def test_advocacy_only_true_for_post_hoc(model, issue_id):
    """Post-hoc explanatory sources are also below the advocacy trust threshold."""
    _add(model, issue_id, SourceRole.POST_HOC_EXPLANATORY)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["advocacy_only"] is True


# ---------------------------------------------------------------------------
# 10. Mixed sources give intermediate trust score
# ---------------------------------------------------------------------------

def test_mixed_sources_intermediate_trust(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE)    # 1.0
    _add(model, issue_id, SourceRole.ADVOCACY)     # 0.3
    result = model.proof_state.compute_and_store(issue_id)
    expected = ProofStateStore.SOURCE_TRUST["operative"] + ProofStateStore.SOURCE_TRUST["advocacy"]
    assert result["trust_weighted_support"] == pytest.approx(expected, abs=0.01)
    assert result["trust_weighted_support"] > ProofStateStore.SOURCE_TRUST["advocacy"]
    assert result["trust_weighted_support"] < 2 * ProofStateStore.SOURCE_TRUST["operative"]


# ---------------------------------------------------------------------------
# 11. Attacking trust weight
# ---------------------------------------------------------------------------

def test_trust_weighted_attack_populated(model, issue_id):
    _add(model, issue_id, SourceRole.OPERATIVE, relation="supports")
    _add(model, issue_id, SourceRole.ADVOCACY, relation="attacks")
    result = model.proof_state.compute_and_store(issue_id)
    assert result["trust_weighted_attack"] == pytest.approx(
        ProofStateStore.SOURCE_TRUST["advocacy"], abs=0.01
    )


# ---------------------------------------------------------------------------
# 12-13. Trust fields survive through get_gaps / get_by_status
# ---------------------------------------------------------------------------

def test_trust_fields_in_get_gaps(model, issue_id):
    _add(model, issue_id, SourceRole.ADVOCACY)  # low-trust → likely insufficient
    model.proof_state.compute_and_store(issue_id)
    gaps = model.proof_state.get_gaps(threshold=0.9)
    assert len(gaps) >= 1
    assert "trust_weighted_support" in gaps[0]
    assert "advocacy_only" in gaps[0]


def test_trust_fields_in_get_by_status(model, issue_id):
    _add(model, issue_id, SourceRole.ADVOCACY)
    model.proof_state.compute_and_store(issue_id)
    # One advocacy assertion → partial status (sufficiency = 0.5 from raw count formula)
    rows = model.proof_state.get_by_status("partial")
    assert len(rows) >= 1
    assert "advocacy_only" in rows[0]


# ---------------------------------------------------------------------------
# 14. Unknown source_role defaults to 0.5
# ---------------------------------------------------------------------------

def test_unknown_source_role_defaults_to_half(model, issue_id):
    _add(model, issue_id, SourceRole.UNKNOWN)
    result = model.proof_state.compute_and_store(issue_id)
    assert result["trust_weighted_support"] == pytest.approx(0.5, abs=0.01)
    # Unknown is above the advocacy_only threshold
    assert result["advocacy_only"] is False


# ---------------------------------------------------------------------------
# 15. Recompute updates advocacy_only
# ---------------------------------------------------------------------------

def test_recompute_updates_advocacy_only(model, issue_id):
    """advocacy_only starts True (advocacy assertion), flips False after operative added."""
    _add(model, issue_id, SourceRole.ADVOCACY)
    result1 = model.proof_state.compute_and_store(issue_id)
    assert result1["advocacy_only"] is True

    _add(model, issue_id, SourceRole.OPERATIVE)
    result2 = model.proof_state.compute_and_store(issue_id)
    assert result2["advocacy_only"] is False
