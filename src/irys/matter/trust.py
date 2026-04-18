"""P0.2 TrustPolicy — single source of truth for eligibility decisions.

Four stores ask the same question five different ways today: hydration,
cached assertion search, proof scoring, coverage reporting, and
synthesis packet assembly each re-derive "is this assertion usable?"
from inline `verification.status != 'rejected'` checks, with no
agreement on what `stale` means or whether the edge's own state
matters. This module collapses that into one canonical precedence
rule so every consumer agrees.

Precedence (highest → lowest):
  excluded : clean-mode privilege filter, inactive belief state
             (disputed/withdrawn/superseded), or any relevant
             verification status is `rejected`.
  stale    : not excluded and any relevant verification status is
             `stale`.
  verified : every relevant verification status is `verified`
             (assertion AND edge when an edge is part of the read).
  candidate: otherwise (default for AI-derived intelligence).

Purposes declare which reads may use which buckets. The policy
returns either (True, bucket) for eligible reads or (False, bucket)
for ineligible ones, letting callers choose between filtering and
labeling.

SO-2 (typed assertion graph): candidate and verified must be visible
as distinct lanes, never collapsed into a single eligibility boolean.
SO-3 (user-steerable reasoning): stale/rejected must never be treated
as equivalent to candidate, because the human reviewer has already
expressed an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .enums import BeliefState


class TrustBucket(str, Enum):
    """The eligibility classification a TrustPolicy assigns to a read."""

    VERIFIED = "verified"
    CANDIDATE = "candidate"
    STALE = "stale"
    EXCLUDED = "excluded"


class TrustPurpose(str, Enum):
    """Why a consumer is reading the assertion — different purposes
    allow different buckets."""

    HYDRATION = "hydration"
    CACHED_SEARCH = "cached_search"
    STRUCTURED_READ = "structured_read"
    PROOF_CANDIDATE = "proof_candidate"
    PROOF_VERIFIED = "proof_verified"
    SYNTHESIS_DEFINITIVE = "synthesis_definitive"


# Belief states that exclude the assertion from any consumer read.
_INACTIVE_BELIEF_STATES: frozenset[str] = frozenset({
    BeliefState.DISPUTED.value,
    BeliefState.WITHDRAWN.value,
    BeliefState.SUPERSEDED.value,
})


_ELIGIBLE_BUCKETS: dict[TrustPurpose, frozenset[TrustBucket]] = {
    # Hydration sees everything so callers can render buckets separately.
    TrustPurpose.HYDRATION: frozenset({
        TrustBucket.VERIFIED, TrustBucket.CANDIDATE,
        TrustBucket.STALE, TrustBucket.EXCLUDED,
    }),
    # Cached assertion search returns leads + verified; stale/excluded
    # are dropped so the UI cannot treat them as usable.
    TrustPurpose.CACHED_SEARCH: frozenset({
        TrustBucket.VERIFIED, TrustBucket.CANDIDATE,
    }),
    # Structured reads for synthesis visibility — same eligibility as
    # cached search. Stale assertions have explicit human history
    # against them and must not re-enter context unchecked.
    TrustPurpose.STRUCTURED_READ: frozenset({
        TrustBucket.VERIFIED, TrustBucket.CANDIDATE,
    }),
    # Advisory/candidate proof lane: verified + candidate counted.
    TrustPurpose.PROOF_CANDIDATE: frozenset({
        TrustBucket.VERIFIED, TrustBucket.CANDIDATE,
    }),
    # Authoritative proof lane: verified-only.
    TrustPurpose.PROOF_VERIFIED: frozenset({TrustBucket.VERIFIED}),
    # Definitive synthesis: verified support only; everything else
    # must be hedged or abstained.
    TrustPurpose.SYNTHESIS_DEFINITIVE: frozenset({TrustBucket.VERIFIED}),
}


@dataclass(frozen=True)
class TrustClassification:
    """Result of classifying one read through TrustPolicy."""

    bucket: TrustBucket
    reason: str  # human-readable why this bucket was chosen
    eligible: bool

    @property
    def is_verified(self) -> bool:
        return self.bucket is TrustBucket.VERIFIED

    @property
    def is_candidate(self) -> bool:
        return self.bucket is TrustBucket.CANDIDATE


class TrustPolicy:
    """Canonical eligibility policy used by hydration, search, proof,
    and synthesis consumers."""

    @staticmethod
    def classify(
        *,
        assertion_verification_status: Optional[str],
        edge_verification_status: Optional[str] = None,
        belief_state: Optional[str] = None,
        privilege_flag: Optional[bool] = None,
        policy_audience: str = "internal",
    ) -> TrustClassification:
        """Classify one read. `edge_verification_status` is None when
        the read does not involve an edge (e.g. hydration of a raw
        assertion). Privilege + audience jointly decide whether the
        read is excluded for the clean lane.

        Unknown statuses are treated as `candidate` (P0.1 default)
        rather than exclusion — AI-derived objects without a
        verification_state row are effectively fresh candidates.
        """
        # Clean-mode privilege exclusion takes precedence. None/True both
        # mean "cannot prove non-privileged" and default to exclusion.
        if policy_audience == "clean":
            if privilege_flag is None or privilege_flag:
                return TrustClassification(
                    bucket=TrustBucket.EXCLUDED,
                    reason="clean_mode_privileged_or_unknown",
                    eligible=False,
                )

        # Belief-state exclusion: disputed/withdrawn/superseded
        # assertions drop out regardless of verification lane.
        if belief_state in _INACTIVE_BELIEF_STATES:
            return TrustClassification(
                bucket=TrustBucket.EXCLUDED,
                reason=f"belief_state_{belief_state}",
                eligible=False,
            )

        # Rejected outranks stale outranks verified outranks candidate.
        statuses = [
            s for s in (assertion_verification_status, edge_verification_status)
            if s is not None
        ]
        if "rejected" in statuses:
            return TrustClassification(
                bucket=TrustBucket.EXCLUDED,
                reason="verification_rejected",
                eligible=False,
            )
        if "stale" in statuses:
            return TrustClassification(
                bucket=TrustBucket.STALE,
                reason="verification_stale",
                eligible=False,
            )
        # Verified requires every relevant lane to be verified. A verified
        # assertion with a candidate edge is only candidate.
        if statuses and all(s == "verified" for s in statuses):
            return TrustClassification(
                bucket=TrustBucket.VERIFIED,
                reason="all_verified",
                eligible=True,
            )
        return TrustClassification(
            bucket=TrustBucket.CANDIDATE,
            reason="candidate_default",
            eligible=True,
        )

    @staticmethod
    def is_eligible(
        bucket: TrustBucket, purpose: TrustPurpose,
    ) -> bool:
        """Check whether a given bucket may be used for a given purpose."""
        return bucket in _ELIGIBLE_BUCKETS[purpose]

    @staticmethod
    def eligible_buckets(purpose: TrustPurpose) -> frozenset[TrustBucket]:
        """The bucket set a caller may use for a purpose."""
        return _ELIGIBLE_BUCKETS[purpose]

    @staticmethod
    def bucket_sql_fragment(
        *,
        purpose: TrustPurpose,
        assertion_status_col: str = "vs.status",
        edge_status_col: Optional[str] = None,
        belief_state_col: Optional[str] = None,
        privilege_col: Optional[str] = None,
        policy_audience: str = "internal",
    ) -> str:
        """Return a parameter-free SQL fragment that evaluates to 1
        when the row is eligible for `purpose`, otherwise 0. Used by
        store reads that want set-based trust filtering without
        per-row Python round-trips.

        Arguments are column references that the caller already has in
        scope. None values mean "this join is absent" (e.g. hydration
        of raw assertions has no edge).
        """
        checks: list[str] = []
        if policy_audience == "clean" and privilege_col is not None:
            # Exclude if flag is 1 or NULL (fail-closed).
            checks.append(
                f"({privilege_col} IS NOT NULL AND {privilege_col}=0)"
            )
        if belief_state_col is not None:
            inactive_list = ",".join(
                f"'{s}'" for s in sorted(_INACTIVE_BELIEF_STATES)
            )
            checks.append(
                f"({belief_state_col} IS NULL OR "
                f"{belief_state_col} NOT IN ({inactive_list}))"
            )
        purpose_buckets = _ELIGIBLE_BUCKETS[purpose]
        # Map bucket requirements to status constraints.
        if TrustBucket.EXCLUDED not in purpose_buckets:
            checks.append(f"({assertion_status_col} != 'rejected' OR {assertion_status_col} IS NULL)")
            if edge_status_col is not None:
                checks.append(f"({edge_status_col} != 'rejected' OR {edge_status_col} IS NULL)")
        if TrustBucket.STALE not in purpose_buckets:
            checks.append(f"({assertion_status_col} != 'stale' OR {assertion_status_col} IS NULL)")
            if edge_status_col is not None:
                checks.append(f"({edge_status_col} != 'stale' OR {edge_status_col} IS NULL)")
        if TrustBucket.CANDIDATE not in purpose_buckets:
            checks.append(f"({assertion_status_col} = 'verified')")
            if edge_status_col is not None:
                checks.append(f"({edge_status_col} = 'verified')")
        if not checks:
            return "1"
        return " AND ".join(checks)


class ContentPurpose(str, Enum):
    """P0.5 Content Policy MVI — the set of user-facing surfaces and
    internal consumers that must route through the policy guard before
    privileged or unknown material can leak into clean-mode output.
    """

    PROFILE = "profile"                           # document profile / card write
    DEEP_READ = "deep_read"                       # per-issue detailed extraction
    SEARCH_SNIPPETS_TO_LLM = "search_snippets_to_llm"
    HYDRATION = "hydration"                       # seed accumulated_facts
    SYNTHESIS_CONTEXT = "synthesis_context"       # packet assembly
    TIMELINE_VIEW = "timeline_view"
    MATRIX_VIEW = "matrix_view"
    CHAT_RESPONSE = "chat_response"
    EXPORT = "export"


class ContentAction(str, Enum):
    """Three outcomes the guard can emit."""

    ALLOW = "allow"
    BLOCK = "block"               # drop silently; no placeholder
    WITHHOLD = "withhold"         # emit "[withheld]" — preserve shape


# Stable machine reason codes used by the audit log and tests.
REASON_ALLOWED_INTERNAL = "allowed_internal_audience"
REASON_ALLOWED_VERIFIED = "allowed_verified"
REASON_ALLOWED_CANDIDATE = "allowed_candidate"
REASON_PRIVILEGE_CLEAN_MODE = "clean_mode_privileged_or_unknown"
REASON_VERIFICATION_REJECTED = "verification_rejected"
REASON_VERIFICATION_STALE = "verification_stale"
REASON_BELIEF_INACTIVE = "belief_state_inactive"


@dataclass(frozen=True)
class ContentPolicyDecision:
    """Result of evaluating a single content-policy request."""

    action: ContentAction
    reason_code: str
    trust_bucket: TrustBucket
    placeholder: Optional[str] = None

    @property
    def is_allowed(self) -> bool:
        return self.action is ContentAction.ALLOW

    @property
    def is_withheld(self) -> bool:
        return self.action is ContentAction.WITHHOLD

    @property
    def is_blocked(self) -> bool:
        return self.action is ContentAction.BLOCK


# Purposes that preserve shape via "[withheld]" placeholder when
# clean-mode privilege excludes the content. All other purposes drop
# the target silently (BLOCK).
_WITHHOLD_PURPOSES: frozenset[ContentPurpose] = frozenset({
    ContentPurpose.TIMELINE_VIEW,
    ContentPurpose.MATRIX_VIEW,
    ContentPurpose.EXPORT,
    ContentPurpose.HYDRATION,              # hydration shows count, not content
    ContentPurpose.SEARCH_SNIPPETS_TO_LLM, # placeholder lets LLM see "gap here"
    ContentPurpose.SYNTHESIS_CONTEXT,      # scrub replaces with placeholder
})


WITHHELD_PLACEHOLDER = "[withheld]"


class ContentPolicy:
    """Single source of truth for every "can this target enter clean
    output for purpose X?" decision in the system.

    Composes TrustPolicy (verification + belief) with the clean/
    internal privilege gate. Returns (action, reason_code,
    trust_bucket, placeholder) so callers can:
      - act on `action` (allow | block | withhold),
      - render `placeholder` when withholding (preserves shape),
      - write `reason_code` into the audit log.
    """

    @staticmethod
    def decide(
        *,
        purpose: ContentPurpose,
        policy_audience: str = "clean",
        assertion_verification_status: Optional[str] = None,
        edge_verification_status: Optional[str] = None,
        belief_state: Optional[str] = None,
        privilege_flag: Optional[bool] = None,
    ) -> ContentPolicyDecision:
        """Evaluate a single content-policy request."""
        # Internal audience bypasses privilege AND verification — the
        # attorney sees everything, including their own privileged
        # work product.
        if policy_audience != "clean":
            return ContentPolicyDecision(
                action=ContentAction.ALLOW,
                reason_code=REASON_ALLOWED_INTERNAL,
                trust_bucket=TrustBucket.VERIFIED,
            )
        # Clean-mode privilege exclusion. None/True both fail closed.
        if privilege_flag is None or privilege_flag:
            if purpose in _WITHHOLD_PURPOSES:
                return ContentPolicyDecision(
                    action=ContentAction.WITHHOLD,
                    reason_code=REASON_PRIVILEGE_CLEAN_MODE,
                    trust_bucket=TrustBucket.EXCLUDED,
                    placeholder=WITHHELD_PLACEHOLDER,
                )
            return ContentPolicyDecision(
                action=ContentAction.BLOCK,
                reason_code=REASON_PRIVILEGE_CLEAN_MODE,
                trust_bucket=TrustBucket.EXCLUDED,
            )
        # Privilege cleared. Now evaluate the trust bucket through
        # the existing TrustPolicy classifier.
        classification = TrustPolicy.classify(
            assertion_verification_status=assertion_verification_status,
            edge_verification_status=edge_verification_status,
            belief_state=belief_state,
            privilege_flag=False,
            policy_audience=policy_audience,
        )
        bucket = classification.bucket
        if bucket is TrustBucket.EXCLUDED:
            reason = (
                REASON_VERIFICATION_REJECTED
                if "rejected" in (classification.reason or "")
                else REASON_BELIEF_INACTIVE
            )
            action = (
                ContentAction.WITHHOLD
                if purpose in _WITHHOLD_PURPOSES
                else ContentAction.BLOCK
            )
            placeholder = WITHHELD_PLACEHOLDER if action is ContentAction.WITHHOLD else None
            return ContentPolicyDecision(
                action=action, reason_code=reason,
                trust_bucket=bucket, placeholder=placeholder,
            )
        if bucket is TrustBucket.STALE:
            action = (
                ContentAction.WITHHOLD
                if purpose in _WITHHOLD_PURPOSES
                else ContentAction.BLOCK
            )
            return ContentPolicyDecision(
                action=action,
                reason_code=REASON_VERIFICATION_STALE,
                trust_bucket=bucket,
                placeholder=(
                    WITHHELD_PLACEHOLDER if action is ContentAction.WITHHOLD else None
                ),
            )
        if bucket is TrustBucket.VERIFIED:
            return ContentPolicyDecision(
                action=ContentAction.ALLOW,
                reason_code=REASON_ALLOWED_VERIFIED,
                trust_bucket=bucket,
            )
        # candidate
        return ContentPolicyDecision(
            action=ContentAction.ALLOW,
            reason_code=REASON_ALLOWED_CANDIDATE,
            trust_bucket=bucket,
        )


__all__ = [
    "TrustBucket",
    "TrustPurpose",
    "TrustPolicy",
    "TrustClassification",
    "ContentPurpose",
    "ContentAction",
    "ContentPolicyDecision",
    "ContentPolicy",
    "WITHHELD_PLACEHOLDER",
    "REASON_ALLOWED_INTERNAL",
    "REASON_ALLOWED_VERIFIED",
    "REASON_ALLOWED_CANDIDATE",
    "REASON_PRIVILEGE_CLEAN_MODE",
    "REASON_VERIFICATION_REJECTED",
    "REASON_VERIFICATION_STALE",
    "REASON_BELIEF_INACTIVE",
]
