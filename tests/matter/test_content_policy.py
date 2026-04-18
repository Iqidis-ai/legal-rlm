"""P0.5 Content Policy MVI unit tests (SO-5).

Covers the single-source-of-truth ContentPolicy.decide():
- clean-mode privilege failures → BLOCK or WITHHOLD by purpose
- internal audience bypass (attorney sees everything)
- verification-rejected + belief-inactive carry through to BLOCK/WITHHOLD
- stale under clean is also excluded (matches TrustPolicy)
- verified + candidate non-privileged → ALLOW
- placeholder is "[withheld]" for the withhold purposes
"""

from irys.matter.enums import BeliefState
from irys.matter.trust import (
    ContentAction,
    ContentPolicy,
    ContentPurpose,
    TrustBucket,
    WITHHELD_PLACEHOLDER,
    REASON_ALLOWED_INTERNAL,
    REASON_ALLOWED_VERIFIED,
    REASON_ALLOWED_CANDIDATE,
    REASON_PRIVILEGE_CLEAN_MODE,
    REASON_VERIFICATION_REJECTED,
    REASON_VERIFICATION_STALE,
    REASON_BELIEF_INACTIVE,
)


# ---------------------------------------------------------------------------
# Internal audience bypass
# ---------------------------------------------------------------------------

def test_internal_audience_always_allows():
    """Attorneys see their own work. Internal audience ignores both
    privilege and verification gates."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        policy_audience="internal",
        privilege_flag=True,
        assertion_verification_status="rejected",
    )
    assert d.is_allowed
    assert d.reason_code == REASON_ALLOWED_INTERNAL


# ---------------------------------------------------------------------------
# Clean-mode privilege exclusion
# ---------------------------------------------------------------------------

def test_clean_privileged_timeline_withholds_with_placeholder():
    """Timeline preserves shape — privileged rows become "[withheld]"
    so the reader sees a gap, not silence."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.action is ContentAction.WITHHOLD
    assert d.reason_code == REASON_PRIVILEGE_CLEAN_MODE
    assert d.placeholder == WITHHELD_PLACEHOLDER


def test_clean_privileged_chat_response_blocks():
    """Chat response drops privileged content silently — the user's
    chat turn doesn't need shape preservation."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.CHAT_RESPONSE,
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.action is ContentAction.BLOCK
    assert d.reason_code == REASON_PRIVILEGE_CLEAN_MODE


def test_clean_unknown_privilege_fails_closed():
    """Privilege unknown is treated identically to True under clean —
    MVP.4 fail-closed contract."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.MATRIX_VIEW,
        policy_audience="clean",
        privilege_flag=None,
    )
    assert d.is_withheld
    assert d.reason_code == REASON_PRIVILEGE_CLEAN_MODE


def test_clean_export_privileged_withholds():
    """Export is a withhold purpose — a clean export preserves the
    row shape with a placeholder."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.EXPORT,
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.is_withheld


def test_clean_profile_privileged_blocks():
    """Profile is not a withhold purpose — privileged docs don't
    get profiled in clean mode at all."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.PROFILE,
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.is_blocked


def test_clean_deep_read_privileged_blocks():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.DEEP_READ,
        policy_audience="clean",
        privilege_flag=True,
    )
    assert d.is_blocked


# ---------------------------------------------------------------------------
# Verification exclusions (post privilege)
# ---------------------------------------------------------------------------

def test_clean_nonpriv_rejected_blocks():
    """Non-privileged but rejected: BLOCK for non-withhold purposes."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.CHAT_RESPONSE,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="rejected",
    )
    assert d.is_blocked
    assert d.reason_code == REASON_VERIFICATION_REJECTED


def test_clean_nonpriv_rejected_withholds_for_shape_purposes():
    """Rejected under a withhold purpose still preserves shape."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="rejected",
    )
    assert d.is_withheld
    assert d.reason_code == REASON_VERIFICATION_REJECTED


def test_clean_nonpriv_stale_excluded():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="stale",
    )
    assert d.is_withheld
    assert d.reason_code == REASON_VERIFICATION_STALE


def test_clean_nonpriv_withdrawn_belief_excluded():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.CHAT_RESPONSE,
        policy_audience="clean",
        privilege_flag=False,
        belief_state=BeliefState.WITHDRAWN.value,
    )
    assert d.is_blocked
    assert d.reason_code == REASON_BELIEF_INACTIVE


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_clean_nonpriv_verified_allows():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="verified",
    )
    assert d.is_allowed
    assert d.reason_code == REASON_ALLOWED_VERIFIED
    assert d.trust_bucket is TrustBucket.VERIFIED


def test_clean_nonpriv_candidate_allows():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="candidate",
    )
    assert d.is_allowed
    assert d.reason_code == REASON_ALLOWED_CANDIDATE
    assert d.trust_bucket is TrustBucket.CANDIDATE


def test_clean_nonpriv_no_verification_row_treated_as_candidate():
    """Missing verification row defaults to candidate (P0.1 contract)."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.HYDRATION,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status=None,
    )
    assert d.is_allowed
    assert d.trust_bucket is TrustBucket.CANDIDATE


# ---------------------------------------------------------------------------
# Edge + assertion lane interaction (reuses TrustPolicy semantics)
# ---------------------------------------------------------------------------

def test_clean_verified_assertion_candidate_edge_is_candidate():
    """Verified requires both lanes to be verified — an edge-backed
    read where the edge is candidate downgrades to candidate lane."""
    d = ContentPolicy.decide(
        purpose=ContentPurpose.SYNTHESIS_CONTEXT,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="verified",
        edge_verification_status="candidate",
    )
    assert d.is_allowed
    assert d.trust_bucket is TrustBucket.CANDIDATE


def test_clean_stale_edge_drops_to_withhold_under_shape_purpose():
    d = ContentPolicy.decide(
        purpose=ContentPurpose.TIMELINE_VIEW,
        policy_audience="clean",
        privilege_flag=False,
        assertion_verification_status="verified",
        edge_verification_status="stale",
    )
    assert d.is_withheld
    assert d.reason_code == REASON_VERIFICATION_STALE
