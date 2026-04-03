"""BeliefRevisionEngine — propagates belief state changes through the assertion graph.

When an assertion changes state (e.g., from unknown → disputed, or from
operative → superseded), dependent assertions must be re-evaluated.

The algorithm:
1. Starting from a set of seed assertion IDs, perform BFS over the
   dependency graph (assertion_link table).
2. For each reached assertion, recompute belief state from its support
   and attack links.
3. Record a belief_revision_event for every assertion whose state changes.
4. Continue until no further changes propagate.

This gives us truth-maintenance: a single user correction can ripple
through the graph and update all downstream conclusions.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

_log = logging.getLogger(__name__)

from .db import SQLiteMatterDB
from .enums import BeliefState, RevisionCause, SOURCE_TRUST_WEIGHTS
from .models import RevisionResult
from .graph import AssertionStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


# Source role trust weights for belief revision (SO-5).
# Canonical definition lives in enums.SOURCE_TRUST_WEIGHTS.
# This alias preserves the existing API for callers and tests that reference _SOURCE_TRUST.
_SOURCE_TRUST = SOURCE_TRUST_WEIGHTS

# Belief state transition rules based on support/attack balance
# A full truth-maintenance system would use JTMS; this is a practical
# approximation sufficient for legal intelligence at current scale.

def _compute_belief_state(
    current_state: BeliefState,
    support_states: list[BeliefState],
    attack_states: list[BeliefState],
    has_superseding: bool = False,
    current_confidence: float = 0.5,
    support_source_roles: "list[str] | None" = None,
    attack_source_roles: "list[str] | None" = None,
) -> tuple[BeliefState, float]:
    """
    Compute new belief state and confidence from support/attack/supersedes graph.

    State transitions are binary (presence of attack/support determines the
    transition). Confidence is trust-weighted: an advocacy-authored attacker
    (weight 0.3) has less impact on confidence than an operative attacker (1.0).

    When source_roles are None, defaults to weight 1.0 for all (backward compatible).

    Returns (new_belief_state, new_confidence).
    """
    # Superseded/Withdrawn are terminal states — graph cannot un-do them
    if current_state == BeliefState.SUPERSEDED:
        return BeliefState.SUPERSEDED, 0.1

    if current_state == BeliefState.WITHDRAWN:
        return BeliefState.WITHDRAWN, 0.0

    # SUPERSEDES link: a newer assertion explicitly replaces this one → terminal
    if has_superseding:
        return BeliefState.SUPERSEDED, 0.1

    # Compute trust weights — 1.0 when no role info (fully backward compatible).
    # Validate list alignment: mismatched lengths would cause zip() to silently
    # truncate the longer list, producing wrong trust weights.
    if support_source_roles is not None and len(support_source_roles) != len(support_states):
        raise ValueError(
            f"support_source_roles length {len(support_source_roles)} != "
            f"support_states length {len(support_states)}"
        )
    if attack_source_roles is not None and len(attack_source_roles) != len(attack_states):
        raise ValueError(
            f"attack_source_roles length {len(attack_source_roles)} != "
            f"attack_states length {len(attack_states)}"
        )
    sup_weights = (
        [_SOURCE_TRUST.get(r, 0.5) for r in support_source_roles]
        if support_source_roles is not None
        else [1.0] * len(support_states)
    )
    atk_weights = (
        [_SOURCE_TRUST.get(r, 0.5) for r in attack_source_roles]
        if attack_source_roles is not None
        else [1.0] * len(attack_states)
    )

    _INERT = (BeliefState.WITHDRAWN, BeliefState.SUPERSEDED, BeliefState.UNKNOWN)
    _UNDERMINING = (BeliefState.DISPUTED, BeliefState.WITHDRAWN,
                    BeliefState.SUPERSEDED, BeliefState.UNKNOWN)

    # Build trust-weighted filtered lists (state + weight pairs → weights only)
    active_attack_weights = [w for s, w in zip(attack_states, atk_weights) if s not in _INERT]
    strong_support_weights = [w for s, w in zip(support_states, sup_weights) if s not in _UNDERMINING]
    operative_support_weights = [
        w for s, w in zip(support_states, sup_weights)
        if s not in _UNDERMINING and s == BeliefState.OPERATIVE
    ]

    if active_attack_weights and not strong_support_weights:
        # Actively attacked with no solid support → disputed
        # Confidence penalty scales with effective attack weight (SO-5: advocacy attacks hurt less)
        return BeliefState.DISPUTED, max(0.1, 0.5 - 0.1 * sum(active_attack_weights))

    if active_attack_weights and strong_support_weights:
        # Both sides present → disputed
        # Confidence reflects the trust-weighted balance: support-dominant → closer to 0.5 (weakly disputed)
        # attack-dominant → closer to 0.3 (strongly disputed). Range: [0.3, 0.5)
        _eff_atk = sum(active_attack_weights)
        _eff_sup = sum(strong_support_weights)
        _total = _eff_atk + _eff_sup
        _sup_frac = _eff_sup / _total if _total > 0 else 0.5
        return BeliefState.DISPUTED, round(0.3 + 0.2 * _sup_frac, 4)

    if support_states and not strong_support_weights and not active_attack_weights:
        # Assertion has supporters, but NONE are solid (all disputed/unknown/superseded/withdrawn)
        # — the support base has collapsed; revert to UNKNOWN
        return BeliefState.UNKNOWN, 0.3

    if operative_support_weights and not active_attack_weights:
        # Solid operative support with no attacks → inferred
        # Confidence boost scales with effective operative weight (SO-5: advocacy-source operative = less boost)
        confidence = min(0.9, 0.5 + 0.1 * sum(operative_support_weights))
        return BeliefState.INFERRED, confidence

    if strong_support_weights and not active_attack_weights:
        # Non-operative but solid support, no attacks → keep current with mild boost
        # Never downgrade below current_confidence — seeded values should not be clobbered
        base_confidence = max(current_confidence, min(0.8, 0.5 + 0.05 * sum(strong_support_weights)))
        return current_state, base_confidence

    # No conclusive signal — keep current state and preserve existing confidence
    return current_state, current_confidence


class BeliefRevisionEngine:
    """
    Propagates belief state changes through the assertion dependency graph.

    All revision events are persisted to belief_revision_event.
    The propagation uses fixpoint convergence: nodes whose state changes
    re-enqueue their dependents, allowing downstream nodes to be recomputed
    even when they were already processed in an earlier pass.  This correctly
    handles converging-evidence graphs where a downstream node depends on
    multiple upstream nodes revised in the same traversal.

    Terminates when no further state changes occur, or when MAX_WORK total
    node-visits have been performed (configurable guardrail against unbounded
    propagation in cyclic graphs).
    """

    MAX_WORK = 500

    def __init__(self, db: SQLiteMatterDB, assertion_store: AssertionStore):
        self.db = db
        self.assertion_store = assertion_store

    def apply(
        self,
        seed_assertion_ids: list[str],
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> list[RevisionResult]:
        """Apply belief revision starting from seed assertions.

        Uses fixpoint propagation: after each state change, all dependents are
        re-enqueued so they can recompute with the updated upstream state.
        Seeds receive one unconditional propagation pass (to handle the case
        where the seed state was force-written externally and its own recomputed
        state does not change, but dependents still need to see the new value).

        Returns all RevisionResult objects for assertions whose state changed.

        **Partial-result behaviour:** if the pending queue is not empty when
        MAX_WORK is reached, propagation is cut short and the returned list
        contains only the revisions completed so far.  A WARNING is logged so
        operators can detect the condition; callers must not assume that an
        empty or non-empty result list means convergence was reached.  In dense
        or cyclic graphs that hit this limit, raise MAX_WORK on the class before
        running the affected matter.
        """
        from collections import deque

        # Pre-fetch trust overrides once for the entire traversal.
        override_rows = self.db.execute(
            """SELECT document_pattern, trust_level FROM document_trust_override
               WHERE matter_id=? AND trust_level != 'normal'
               ORDER BY LENGTH(document_pattern) DESC""",
            (self.assertion_store.matter_id,),
        ).fetchall()
        override_cache: list[tuple[str, str]] = [
            (r["document_pattern"], r["trust_level"]) for r in override_rows
        ]

        results: list[RevisionResult] = []
        seeds_remaining: set[str] = set(seed_assertion_ids)
        pending: deque[str] = deque(seed_assertion_ids)
        # in_queue: prevents the same node from being added to pending twice
        # concurrently.  Discarded when a node is dequeued so later state
        # changes can re-enqueue it.
        in_queue: set[str] = set(seed_assertion_ids)
        total_work: int = 0

        while pending and total_work < self.MAX_WORK:
            assertion_id = pending.popleft()
            in_queue.discard(assertion_id)
            is_seed = assertion_id in seeds_remaining
            seeds_remaining.discard(assertion_id)
            total_work += 1

            result = self._revise_one(assertion_id, cause, run_id, note, override_cache)
            if result is not None:
                results.append(result)

            # Propagate to dependents when:
            # - state changed: downstream nodes must recompute with the new upstream value
            # - seed first-pass: propagate once even without change (the seed's state was
            #   force-written externally; its dependents must still see it)
            if result is not None or is_seed:
                for d in self.assertion_store.get_dependents(assertion_id):
                    if d not in in_queue:
                        in_queue.add(d)
                        pending.append(d)

        if pending:
            _log.warning(
                "BeliefRevisionEngine.apply() reached MAX_WORK=%d; %d nodes remain "
                "unprocessed — propagation may be incomplete in dense/cyclic graphs. "
                "Consider raising MAX_WORK if this matter is known to have deep "
                "dependency chains.",
                self.MAX_WORK,
                len(pending),
            )

        return results

    def _revise_one(
        self,
        assertion_id: str,
        cause: RevisionCause,
        run_id: Optional[str],
        note: Optional[str],
        _override_cache: "list[tuple[str, str]] | None" = None,
    ) -> Optional[RevisionResult]:
        """
        Revise a single assertion's belief state based on its graph neighbors.

        Returns RevisionResult if the state changed, None otherwise.
        """
        record = self.assertion_store.get(assertion_id)
        if record is None:
            return None

        old_state = BeliefState(record.belief_state)
        old_confidence = record.confidence

        # Batch-load all neighbor belief states in one query (avoids N+1)
        neighbors = self.assertion_store.get_neighbor_belief_states(
            assertion_id, _override_cache=_override_cache
        )

        new_state, new_confidence = _compute_belief_state(
            old_state,
            neighbors["support_states"],
            neighbors["attack_states"],
            has_superseding=neighbors["has_superseding"],
            current_confidence=old_confidence,
            support_source_roles=neighbors["support_source_roles"],
            attack_source_roles=neighbors["attack_source_roles"],
        )

        if new_state == old_state and abs(new_confidence - old_confidence) < 0.01:
            return None  # No change

        # Persist the state change
        now = _now()
        with self.db.transaction():
            self.assertion_store.set_belief_state(assertion_id, new_state, new_confidence)
            self.db.execute(
                """INSERT INTO belief_revision_event
                   (id, assertion_id, run_id, cause,
                    old_belief_state, new_belief_state,
                    old_confidence, new_confidence, note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    _id(), assertion_id, run_id,
                    cause.value,
                    old_state.value, new_state.value,
                    old_confidence, new_confidence,
                    note, now,
                ),
            )

        return RevisionResult(
            assertion_id=assertion_id,
            old_belief_state=old_state,
            new_belief_state=new_state,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            cause=cause,
        )

    def force_state(
        self,
        assertion_id: str,
        new_state: BeliefState,
        new_confidence: float,
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> RevisionResult:
        """
        Forcibly set an assertion's belief state (e.g., from user correction).
        Then propagate to dependents.
        """
        record = self.assertion_store.get(assertion_id)
        if record is None:
            raise ValueError(f"Assertion {assertion_id} not found")

        old_state = BeliefState(record.belief_state)
        old_confidence = record.confidence
        now = _now()

        with self.db.transaction():
            self.assertion_store.set_belief_state(assertion_id, new_state, new_confidence)
            self.db.execute(
                """INSERT INTO belief_revision_event
                   (id, assertion_id, run_id, cause,
                    old_belief_state, new_belief_state,
                    old_confidence, new_confidence, note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    _id(), assertion_id, run_id,
                    cause.value,
                    old_state.value, new_state.value,
                    old_confidence, new_confidence,
                    note, now,
                ),
            )

        result = RevisionResult(
            assertion_id=assertion_id,
            old_belief_state=old_state,
            new_belief_state=new_state,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            cause=cause,
        )

        # Propagate to dependents
        dependents = self.assertion_store.get_dependents(assertion_id)
        if dependents:
            downstream = self.apply(dependents, cause, run_id, note)
            result.propagated_to = [r.assertion_id for r in downstream]

        return result
