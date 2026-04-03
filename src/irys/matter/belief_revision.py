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

import uuid
from datetime import datetime, timezone
from typing import Optional

from .db import SQLiteMatterDB
from .enums import BeliefState, AssertionLinkType, RevisionCause
from .models import RevisionResult
from .graph import AssertionStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


# Belief state transition rules based on support/attack balance
# A full truth-maintenance system would use JTMS; this is a practical
# approximation sufficient for legal intelligence at current scale.

def _compute_belief_state(
    current_state: BeliefState,
    support_states: list[BeliefState],
    attack_states: list[BeliefState],
    has_superseding: bool = False,
) -> tuple[BeliefState, float]:
    """
    Compute new belief state and confidence from support/attack/supersedes graph.

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

    _INERT = (BeliefState.WITHDRAWN, BeliefState.SUPERSEDED, BeliefState.UNKNOWN)

    # Active attacks: states that genuinely challenge the assertion
    active_attacks = [s for s in attack_states if s not in _INERT]

    # Strong supports: states that genuinely reinforce the assertion.
    # DISPUTED/UNKNOWN/WITHDRAWN/SUPERSEDED supports do not count as solid backing.
    _UNDERMINING = (BeliefState.DISPUTED, BeliefState.WITHDRAWN,
                    BeliefState.SUPERSEDED, BeliefState.UNKNOWN)
    strong_supports = [s for s in support_states if s not in _UNDERMINING]
    operative_supports = [s for s in strong_supports if s == BeliefState.OPERATIVE]

    if active_attacks and not strong_supports:
        # Actively attacked with no solid support → disputed
        return BeliefState.DISPUTED, max(0.1, 0.5 - 0.1 * len(active_attacks))

    if active_attacks and strong_supports:
        # Both sides present → disputed
        return BeliefState.DISPUTED, 0.4

    if support_states and not strong_supports and not active_attacks:
        # Assertion has supporters, but NONE are solid (all disputed/unknown/superseded/withdrawn)
        # — the support base has collapsed; revert to UNKNOWN
        return BeliefState.UNKNOWN, 0.3

    if operative_supports and not active_attacks:
        # Solid operative support with no attacks → inferred
        confidence = min(0.9, 0.5 + 0.1 * len(operative_supports))
        return BeliefState.INFERRED, confidence

    if strong_supports and not active_attacks:
        # Non-operative but solid support, no attacks → keep current with mild boost
        base_confidence = min(0.8, 0.5 + 0.05 * len(strong_supports))
        return current_state, base_confidence

    # No conclusive signal — keep current state unchanged
    return current_state, 0.5


class BeliefRevisionEngine:
    """
    Propagates belief state changes through the assertion dependency graph.

    All revision events are persisted to belief_revision_event.
    The propagation is BFS-limited to max_hops to prevent runaway cascades.
    """

    MAX_HOPS = 10

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
        """
        Apply belief revision starting from seed assertions.

        Propagates through the dependency graph up to MAX_HOPS levels.
        Returns all RevisionResult objects for changed assertions.
        """
        results: list[RevisionResult] = []
        visited: set[str] = set()
        queue = list(seed_assertion_ids)
        hop = 0

        while queue and hop < self.MAX_HOPS:
            next_queue = []
            for assertion_id in queue:
                if assertion_id in visited:
                    continue
                visited.add(assertion_id)

                result = self._revise_one(assertion_id, cause, run_id, note)
                if result is not None:
                    results.append(result)
                    # Propagate to dependents if this assertion changed
                    dependents = self.assertion_store.get_dependents(assertion_id)
                    next_queue.extend(
                        d for d in dependents if d not in visited
                    )

            queue = next_queue
            hop += 1

        return results

    def _revise_one(
        self,
        assertion_id: str,
        cause: RevisionCause,
        run_id: Optional[str],
        note: Optional[str],
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

        # Get support, attack, and supersedes states
        support_ids = self.assertion_store.get_supports(assertion_id)
        attack_ids = self.assertion_store.get_attackers(assertion_id)
        superseding_ids = self.assertion_store.get_superseding(assertion_id)

        support_states = []
        for sid in support_ids:
            r = self.assertion_store.get(sid)
            if r:
                support_states.append(BeliefState(r.belief_state))

        attack_states = []
        for aid in attack_ids:
            r = self.assertion_store.get(aid)
            if r:
                attack_states.append(BeliefState(r.belief_state))

        new_state, new_confidence = _compute_belief_state(
            old_state, support_states, attack_states,
            has_superseding=bool(superseding_ids),
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
