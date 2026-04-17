"""P0.2 TrustPolicy unit tests — one canonical eligibility rule.

Covers:
- precedence (excluded > stale > verified > candidate)
- edge + assertion verification combination (assertion verified +
  edge candidate = candidate, not verified)
- clean-mode privilege behavior (None + True both exclude)
- per-purpose eligibility (e.g. SYNTHESIS_DEFINITIVE allows only
  verified; CACHED_SEARCH drops stale)
"""

from irys.matter.enums import BeliefState
from irys.matter.trust import (
    TrustBucket,
    TrustClassification,
    TrustPolicy,
    TrustPurpose,
)


# ---------------------------------------------------------------------------
# classify() precedence
# ---------------------------------------------------------------------------

def test_classify_rejected_overrides_verified():
    """Rejected is always excluded, even when the other lane is verified."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        edge_verification_status="rejected",
    )
    assert c.bucket is TrustBucket.EXCLUDED
    assert c.reason == "verification_rejected"
    assert c.eligible is False


def test_classify_stale_beats_verified_when_lane_disagrees():
    """Stale on any lane drops to stale bucket (not eligible for most
    purposes). Verified-everywhere is the only way into the verified
    bucket."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        edge_verification_status="stale",
    )
    assert c.bucket is TrustBucket.STALE
    assert c.eligible is False


def test_classify_all_verified_required_for_verified_bucket():
    """Verified requires every relevant lane to be verified."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        edge_verification_status="candidate",
    )
    assert c.bucket is TrustBucket.CANDIDATE


def test_classify_all_verified_yields_verified():
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        edge_verification_status="verified",
    )
    assert c.bucket is TrustBucket.VERIFIED
    assert c.eligible is True


def test_classify_candidate_default_for_ai_derived_intelligence():
    """AI-derived intelligence without a verification_state row is
    classified as candidate (P0.1 default), not excluded."""
    c = TrustPolicy.classify(assertion_verification_status=None)
    assert c.bucket is TrustBucket.CANDIDATE
    assert c.eligible is True


# ---------------------------------------------------------------------------
# Belief-state exclusion
# ---------------------------------------------------------------------------

def test_classify_withdrawn_excluded():
    """Withdrawn beliefs drop out regardless of verification status."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        belief_state=BeliefState.WITHDRAWN.value,
    )
    assert c.bucket is TrustBucket.EXCLUDED
    assert "withdrawn" in c.reason


def test_classify_disputed_excluded():
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        belief_state=BeliefState.DISPUTED.value,
    )
    assert c.bucket is TrustBucket.EXCLUDED


def test_classify_superseded_excluded():
    c = TrustPolicy.classify(
        assertion_verification_status="candidate",
        belief_state=BeliefState.SUPERSEDED.value,
    )
    assert c.bucket is TrustBucket.EXCLUDED


def test_classify_operative_passes_belief_check():
    """Active belief states (operative, alleged, admitted) stay
    inside the verification pipeline."""
    c = TrustPolicy.classify(
        assertion_verification_status="candidate",
        belief_state=BeliefState.OPERATIVE.value,
    )
    assert c.bucket is TrustBucket.CANDIDATE


# ---------------------------------------------------------------------------
# Clean-mode privilege exclusion
# ---------------------------------------------------------------------------

def test_classify_clean_mode_privilege_unknown_excluded():
    """Unknown privilege flag fails closed under clean audience."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        privilege_flag=None,
        policy_audience="clean",
    )
    assert c.bucket is TrustBucket.EXCLUDED
    assert "clean_mode_privileged_or_unknown" in c.reason


def test_classify_clean_mode_privilege_true_excluded():
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        privilege_flag=True,
        policy_audience="clean",
    )
    assert c.bucket is TrustBucket.EXCLUDED


def test_classify_clean_mode_non_privileged_passes():
    """Clean-mode read of an unambiguously non-privileged doc stays
    inside the verification pipeline."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        privilege_flag=False,
        policy_audience="clean",
    )
    assert c.bucket is TrustBucket.VERIFIED


def test_classify_internal_mode_ignores_privilege():
    """Internal audience doesn't apply the privilege exclusion —
    attorneys can still see their own work."""
    c = TrustPolicy.classify(
        assertion_verification_status="verified",
        privilege_flag=True,
        policy_audience="internal",
    )
    assert c.bucket is TrustBucket.VERIFIED


# ---------------------------------------------------------------------------
# Purpose eligibility
# ---------------------------------------------------------------------------

def test_synthesis_definitive_allows_only_verified():
    """Definitive synthesis claims require verified support."""
    assert TrustPolicy.is_eligible(
        TrustBucket.VERIFIED, TrustPurpose.SYNTHESIS_DEFINITIVE,
    )
    assert not TrustPolicy.is_eligible(
        TrustBucket.CANDIDATE, TrustPurpose.SYNTHESIS_DEFINITIVE,
    )
    assert not TrustPolicy.is_eligible(
        TrustBucket.STALE, TrustPurpose.SYNTHESIS_DEFINITIVE,
    )


def test_cached_search_drops_stale_and_excluded():
    """Cached assertion search returns verified + candidate leads."""
    assert TrustPolicy.is_eligible(
        TrustBucket.VERIFIED, TrustPurpose.CACHED_SEARCH,
    )
    assert TrustPolicy.is_eligible(
        TrustBucket.CANDIDATE, TrustPurpose.CACHED_SEARCH,
    )
    assert not TrustPolicy.is_eligible(
        TrustBucket.STALE, TrustPurpose.CACHED_SEARCH,
    )
    assert not TrustPolicy.is_eligible(
        TrustBucket.EXCLUDED, TrustPurpose.CACHED_SEARCH,
    )


def test_hydration_sees_all_buckets():
    """Hydration must see every bucket so callers can render the
    buckets separately — the partitioning is done by the engine, not
    by filtering at the store layer."""
    for bucket in TrustBucket:
        assert TrustPolicy.is_eligible(bucket, TrustPurpose.HYDRATION)


def test_proof_verified_only_accepts_verified():
    assert TrustPolicy.is_eligible(
        TrustBucket.VERIFIED, TrustPurpose.PROOF_VERIFIED,
    )
    for bucket in (
        TrustBucket.CANDIDATE, TrustBucket.STALE, TrustBucket.EXCLUDED,
    ):
        assert not TrustPolicy.is_eligible(
            bucket, TrustPurpose.PROOF_VERIFIED,
        )


def test_proof_candidate_accepts_verified_and_candidate():
    """Advisory proof lane counts verified + candidate; stale drops."""
    assert TrustPolicy.is_eligible(
        TrustBucket.VERIFIED, TrustPurpose.PROOF_CANDIDATE,
    )
    assert TrustPolicy.is_eligible(
        TrustBucket.CANDIDATE, TrustPurpose.PROOF_CANDIDATE,
    )
    assert not TrustPolicy.is_eligible(
        TrustBucket.STALE, TrustPurpose.PROOF_CANDIDATE,
    )


# ---------------------------------------------------------------------------
# SQL-fragment helpers
# ---------------------------------------------------------------------------

def test_bucket_sql_fragment_rejects_rejected_for_cached_search():
    frag = TrustPolicy.bucket_sql_fragment(
        purpose=TrustPurpose.CACHED_SEARCH,
        assertion_status_col="vs.status",
    )
    assert "vs.status != 'rejected'" in frag
    assert "vs.status != 'stale'" in frag


def test_bucket_sql_fragment_includes_privilege_when_clean():
    frag = TrustPolicy.bucket_sql_fragment(
        purpose=TrustPurpose.CACHED_SEARCH,
        assertion_status_col="vs.status",
        privilege_col="d.privilege_flag",
        policy_audience="clean",
    )
    assert "d.privilege_flag" in frag


def test_bucket_sql_fragment_proof_verified_requires_verified():
    frag = TrustPolicy.bucket_sql_fragment(
        purpose=TrustPurpose.PROOF_VERIFIED,
        assertion_status_col="vs.status",
    )
    assert "vs.status = 'verified'" in frag


def test_bucket_sql_fragment_belief_state_excluded():
    frag = TrustPolicy.bucket_sql_fragment(
        purpose=TrustPurpose.CACHED_SEARCH,
        assertion_status_col="vs.status",
        belief_state_col="a.belief_state",
    )
    for state in ("disputed", "superseded", "withdrawn"):
        assert state in frag
