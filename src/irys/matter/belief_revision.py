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

import json as _json_mod
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

_log = logging.getLogger(__name__)

from .db import SQLiteMatterDB
from .enums import BeliefState, LedgerEventType, RevisionCause, SOURCE_TRUST_WEIGHTS
from .models import RevisionResult
from .graph import AssertionStore

if TYPE_CHECKING:  # pragma: no cover
    from .reasoning import ReasoningLedgerStore


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
    superseding_states: "list[BeliefState] | None" = None,
) -> tuple[BeliefState, float]:
    """
    Compute new belief state and confidence from support/attack/supersedes graph.

    State transitions are binary (presence of attack/support determines the
    transition). Confidence is trust-weighted: an advocacy-authored attacker
    (weight 0.3) has less impact on confidence than an operative attacker (1.0).

    When source_roles are None, defaults to weight 1.0 for all (backward compatible).

    Returns (new_belief_state, new_confidence).
    """
    # WITHDRAWN is a truly terminal user-initiated state: the speaker retracted the statement.
    # No graph signal can recover it — only an explicit user correction can.
    if current_state == BeliefState.WITHDRAWN:
        return BeliefState.WITHDRAWN, 0.0

    # SUPERSEDES link handling — two cases depending on whether the caller passes the new
    # superseding_states list (rich interface) or the legacy has_superseding bool.
    #
    # Legacy interface (superseding_states is None):
    #   - SUPERSEDED is terminal regardless (same behavior as before this fix).
    #   - has_superseding=True → force SUPERSEDED.
    #
    # Rich interface (superseding_states provided — used by BFS via get_neighbor_belief_states):
    #   - Active superseder (not WITHDRAWN/SUPERSEDED itself) → force SUPERSEDED.
    #   - No superseding links at all → SUPERSEDED was manually set → keep as terminal.
    #   - Superseding links all inert → superseder was withdrawn or itself superseded →
    #     allow recovery by falling through to normal support/attack logic below.
    #   (HIGH #1 supersession-recovery fix)
    if superseding_states is not None:
        _INERT_SUPERSEDER = (BeliefState.WITHDRAWN, BeliefState.SUPERSEDED)
        _active_superseder = any(s not in _INERT_SUPERSEDER for s in superseding_states)
        _any_superseding_link = bool(superseding_states)
        if _active_superseder:
            return BeliefState.SUPERSEDED, 0.1
        if current_state == BeliefState.SUPERSEDED and not _any_superseding_link:
            # No superseding links at all: SUPERSEDED was manually set (e.g. via
            # correct_assertion without a superseding link). Only an explicit user
            # correction can un-supersede it — graph support cannot.
            return BeliefState.SUPERSEDED, 0.1
        # has_inert_link case: _revise_one() already overrode current_state to the
        # speech-act-derived recovery baseline before calling _compute_belief_state().
        # Fall through to normal support/attack logic with the new baseline state.
    else:
        # Legacy bool interface: preserve original terminal semantics.
        if current_state == BeliefState.SUPERSEDED:
            return BeliefState.SUPERSEDED, 0.1
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

    # Legally authoritative states that promote a dependent to INFERRED.
    # OPERATIVE, ADMITTED, RESOLVED, PERFORMED are all conclusive: a downstream
    # assertion supported by any of these should be elevated to INFERRED.
    # Previously only OPERATIVE was included, so ADMITTED/RESOLVED support left
    # dependents stuck at UNKNOWN/DISPUTED despite legally strong upstream evidence.
    # (HIGH #2 promoting-states fix)
    _PROMOTING = (
        BeliefState.OPERATIVE,
        BeliefState.ADMITTED,
        BeliefState.RESOLVED,
        BeliefState.PERFORMED,
    )

    # Build trust-weighted filtered lists (state + weight pairs → weights only)
    active_attack_weights = [w for s, w in zip(attack_states, atk_weights) if s not in _INERT]
    strong_support_weights = [w for s, w in zip(support_states, sup_weights) if s not in _UNDERMINING]
    promoting_support_weights = [
        w for s, w in zip(support_states, sup_weights)
        if s not in _UNDERMINING and s in _PROMOTING
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

    if promoting_support_weights and not active_attack_weights:
        # Legally conclusive support (operative/admitted/resolved/performed) with no attacks → inferred
        # Confidence boost scales with effective promoting weight (SO-5: advocacy-source = less boost)
        confidence = min(0.9, 0.5 + 0.1 * sum(promoting_support_weights))
        return BeliefState.INFERRED, confidence

    if strong_support_weights and not active_attack_weights:
        # Solid but non-promoting support (e.g. ALLEGED, ARGUED, INFERRED), no attacks
        # → keep current state with mild confidence boost
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

    def __init__(
        self,
        db: SQLiteMatterDB,
        assertion_store: AssertionStore,
        ledger: "Optional[ReasoningLedgerStore]" = None,
    ):
        self.db = db
        self.assertion_store = assertion_store
        self._ledger = ledger

    def _apply_with_truncation(
        self,
        seed_assertion_ids: list[str],
        cause: RevisionCause,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> tuple[list[RevisionResult], bool]:
        """Internal BFS driver. Returns (results, truncated).

        Same as apply() but also returns whether propagation was cut short
        by MAX_WORK, so force_state() can set propagation_truncated on the
        returned RevisionResult without changing the public apply() signature.
        """
        from collections import deque

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
        # Deduplicate seeds before building the deque to prevent duplicate _revise_one() work.
        _seen_seeds: set[str] = set()
        _deduped_seeds: list[str] = []
        for _s in seed_assertion_ids:
            if _s not in _seen_seeds:
                _seen_seeds.add(_s)
                _deduped_seeds.append(_s)
        pending: deque[str] = deque(_deduped_seeds)
        in_queue: set[str] = set(_deduped_seeds)
        total_work: int = 0
        occ_conflict_count: int = 0
        occ_exhausted_count: int = 0
        # Per-node OCC retry budget: when a node OCC-aborts, re-enqueue it (not
        # its dependents) for one more attempt with the latest committed state.
        # Cap retries to prevent BFS explosion under sustained concurrent contention.
        _OCC_MAX_RETRIES: int = 3
        _occ_retries: dict[str, int] = {}
        # Frontier-aware work budget: scale up when seed count is large enough that
        # seeds alone would consume more than half the budget — leaving insufficient
        # room for propagation.  Only activates when needed (preserves MAX_WORK as a
        # hard cap in tests and for small seed counts).  Ceiling is 2000.
        _n_seeds = len(_deduped_seeds)
        _effective_max_work: int = (
            min(2000, max(self.MAX_WORK, _n_seeds * 3))
            if _n_seeds * 2 > self.MAX_WORK
            else self.MAX_WORK
        )

        while pending and total_work < _effective_max_work:
            assertion_id = pending.popleft()
            in_queue.discard(assertion_id)
            is_seed = assertion_id in seeds_remaining
            # Do NOT discard from seeds_remaining here; only consume seedness
            # after a non-aborted pass so OCC retries preserve the seed fan-out.
            total_work += 1

            result, occ_aborted = self._revise_one(assertion_id, cause, run_id, note, override_cache)
            if occ_aborted:
                occ_conflict_count += 1
                # Re-enqueue X itself so BFS retries with the latest committed state.
                # Do NOT enqueue dependents here: we don't know X's new committed state
                # yet, so pre-emptively fanning out would inflate BFS work.  Dependents
                # will be enqueued after X is successfully processed.
                # Do NOT discard from seeds_remaining: seedness must survive OCC retries.
                retries = _occ_retries.get(assertion_id, 0) + 1
                _occ_retries[assertion_id] = retries
                if retries <= _OCC_MAX_RETRIES and assertion_id not in in_queue:
                    in_queue.add(assertion_id)
                    pending.append(assertion_id)
                else:
                    # Retry cap exhausted — node abandoned; subtree and any pending
                    # seed fan-out for this node may be stale.
                    occ_exhausted_count += 1
                continue

            # Non-aborted: consume seedness now.
            seeds_remaining.discard(assertion_id)

            if result is not None:
                results.append(result)

            # Enqueue dependents when state changed or this is an unconditional seed.
            if result is not None or is_seed:
                for d in self.assertion_store.get_dependents(assertion_id):
                    if d not in in_queue:
                        in_queue.add(d)
                        pending.append(d)

        if occ_exhausted_count:
            _log.warning(
                "BeliefRevisionEngine: %d node(s) abandoned after %d OCC retries — "
                "those subtrees may be stale; a follow-up propagation pass is needed.",
                occ_exhausted_count,
                _OCC_MAX_RETRIES,
            )

        if occ_conflict_count:
            _log.warning(
                "BeliefRevisionEngine: %d OCC conflict(s) during propagation "
                "(concurrent writes committed between BFS read and write lock). "
                "Conflicted nodes were re-enqueued for retry (up to %d retries each).",
                occ_conflict_count,
                _OCC_MAX_RETRIES,
            )
            if run_id and self._ledger is not None:
                try:
                    self._ledger.append_event(
                        run_id=run_id,
                        event_type=LedgerEventType.SYSTEM_WARNING,
                        summary=(
                            f"Belief revision: {occ_conflict_count} OCC conflict(s) — "
                            "concurrent writes detected during BFS; conflicted nodes "
                            f"re-enqueued for retry (cap={_OCC_MAX_RETRIES})"
                            + (
                                f"; {occ_exhausted_count} node(s) abandoned after "
                                "retry cap exhaustion — subtrees may be stale"
                                if occ_exhausted_count else ""
                            )
                            + "."
                        ),
                    )
                except Exception as exc:
                    _log.warning("Failed to record OCC ledger event: %s", exc, exc_info=True)

        truncated = bool(pending) or occ_exhausted_count > 0
        if truncated:
            _reasons = []
            if pending:
                _reasons.append(f"work budget={_effective_max_work} reached with {len(pending)} nodes remaining")
            if occ_exhausted_count:
                _reasons.append(f"{occ_exhausted_count} node(s) abandoned after OCC retry cap ({_OCC_MAX_RETRIES})")
            _log.warning(
                "BeliefRevisionEngine: propagation incomplete — %s. "
                "Downstream belief states may be stale.",
                "; ".join(_reasons),
            )
            if run_id and self._ledger is not None:
                try:
                    self._ledger.append_event(
                        run_id=run_id,
                        event_type=LedgerEventType.SYSTEM_WARNING,
                        summary=(
                            f"Belief revision truncated: {'; '.join(_reasons)}. "
                            "Downstream belief states may be stale."
                        ),
                    )
                except Exception as exc:
                    _log.warning("Failed to record truncation ledger event: %s", exc, exc_info=True)

        return results, truncated

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

        See also _apply_with_truncation() for a version that returns a
        (results, truncated) tuple — used internally by force_state() to
        surface propagation completeness in RevisionResult.propagation_truncated.
        """
        results, _ = self._apply_with_truncation(seed_assertion_ids, cause, run_id, note)
        return results

    def _revise_one(
        self,
        assertion_id: str,
        cause: RevisionCause,
        run_id: Optional[str],
        note: Optional[str],
        _override_cache: "list[tuple[str, str]] | None" = None,
    ) -> "tuple[Optional[RevisionResult], bool]":
        """
        Revise a single assertion's belief state based on its graph neighbors.

        Returns (result, occ_aborted):
        - result: RevisionResult if state changed, None if no change or row missing.
        - occ_aborted: True if an OCC conflict caused the write to be skipped.
          BFS callers MUST enqueue dependents even when occ_aborted=True so the
          downstream subtree is not silently pruned.
        """
        record = self.assertion_store.get(assertion_id)
        if record is None:
            return None, False

        old_state = BeliefState(record.belief_state)
        old_confidence = record.confidence

        # Batch-load all neighbor belief states in one query (avoids N+1)
        neighbors = self.assertion_store.get_neighbor_belief_states(
            assertion_id, _override_cache=_override_cache
        )

        # Supersession recovery pre-processing (HIGH #1 fix — second pass):
        # When an assertion is currently SUPERSEDED and all its superseding links are
        # now inert (WITHDRAWN or SUPERSEDED themselves), the superseder has been
        # invalidated and the original assertion may recover.
        #
        # Two guard checks before allowing recovery:
        # (1) User-lock: if the most recent field change to 'belief_state' in
        #     assertion_revision was made by a user (actor_kind='user') setting it
        #     to SUPERSEDED, the user explicitly chose that state. Respect it; do not
        #     recover automatically.
        # (2) Baseline state: reset old_state to the speech-act-derived initial state
        #     (e.g., OPERATIVE, ADMITTED, ALLEGED) so that _compute_belief_state sees
        #     the "natural" pre-supersession baseline rather than SUPERSEDED, and can
        #     correctly compute the recovered state from support/attack evidence.
        # Supersession recovery pre-processing (HIGH #1 fix — second pass):
        # When an assertion is currently SUPERSEDED and all its superseding links are
        # now inert (WITHDRAWN or SUPERSEDED themselves), the superseder has been
        # invalidated and the original assertion may recover.
        #
        # Use _compute_state/_compute_conf as the INPUT to _compute_belief_state so the
        # computation sees the speech-act recovery baseline — NOT the current SUPERSEDED.
        # Keep old_state/old_confidence as the ACTUAL DB state for the fast-path and OCC
        # comparison (we need to detect DB_state=SUPERSEDED → new_state=UNKNOWN as a real
        # change, not a no-op, even when new_state == recovery_baseline).
        _compute_state = old_state
        _compute_conf = old_confidence
        _superseding = neighbors.get("superseding_states") or []
        _INERT_S = (BeliefState.WITHDRAWN, BeliefState.SUPERSEDED)
        _recovery_case = (
            old_state == BeliefState.SUPERSEDED
            and bool(_superseding)
            and not any(s not in _INERT_S for s in _superseding)
        )
        if _recovery_case:
            # Guard 1: user-lock check
            # If the most recent user-authored revision set belief_state to SUPERSEDED,
            # respect user intent and skip automatic recovery.
            _user_lock_row = self.db.execute(
                """SELECT actor_kind FROM assertion_revision
                   WHERE assertion_id=? AND changed_field='belief_state'
                     AND new_value_json=?
                   ORDER BY created_at DESC LIMIT 1""",
                (assertion_id, '"superseded"'),
            ).fetchone()
            if _user_lock_row and _user_lock_row["actor_kind"] == "user":
                return None, False

            # Guard 2: derive speech-act baseline state for the computation.
            # Use the first-occurrence speech_act to reflect the assertion's "natural"
            # state before any supersession was applied. This ensures that an OPERATIVE
            # clause recovers to OPERATIVE (not UNKNOWN) when the superseding amendment
            # is voided. Inlined from _initial_belief_state() in graph.py.
            _occ_row = self.db.execute(
                """SELECT speech_act FROM assertion_occurrence
                   WHERE assertion_id=? ORDER BY created_at ASC LIMIT 1""",
                (assertion_id,),
            ).fetchone()
            _sa = _occ_row["speech_act"] if _occ_row else None
            if _sa == "operative":
                _compute_state, _compute_conf = BeliefState.OPERATIVE, 0.8
            elif _sa in ("admitted", "stipulated"):
                _compute_state, _compute_conf = BeliefState.ADMITTED, 0.8
            elif _sa in ("performed", "paid"):
                _compute_state, _compute_conf = BeliefState.PERFORMED, 0.8
            elif _sa == "inferred":
                _compute_state, _compute_conf = BeliefState.INFERRED, 0.6
            elif _sa == "alleged":
                _compute_state, _compute_conf = BeliefState.ALLEGED, 0.3
            elif _sa == "argued":
                _compute_state, _compute_conf = BeliefState.ARGUED, 0.3
            elif _sa in ("waived", "terminated", "amended"):
                _compute_state, _compute_conf = BeliefState.OPERATIVE, 0.7
            else:
                _compute_state, _compute_conf = BeliefState.UNKNOWN, 0.3

        new_state, new_confidence = _compute_belief_state(
            _compute_state,
            neighbors["support_states"],
            neighbors["attack_states"],
            has_superseding=neighbors["has_superseding"],
            current_confidence=_compute_conf,
            support_source_roles=neighbors["support_source_roles"],
            attack_source_roles=neighbors["attack_source_roles"],
            superseding_states=neighbors.get("superseding_states"),
        )

        # Performance fast-path: skip BEGIN IMMEDIATE when the pre-tx snapshot
        # indicates no state change is needed.  This is safe because:
        # (a) If new_state == old_state, there is nothing to write regardless of
        #     whether a concurrent writer updated the row in the meantime.
        # (b) If a concurrent writer did update old_state, their own BFS propagation
        #     covers dependents — we do not need to act.
        # The authoritative in-tx no-change check still runs for cases where the
        # pre-tx read shows a change but the in-tx re-read reveals a no-op.
        if new_state == old_state and abs(new_confidence - old_confidence) < 0.001:
            return None, False

        # _actual_old_* will be overridden with the committed in-tx values so that the
        # returned RevisionResult accurately reflects what was recorded in the audit trail.
        _actual_old_state: BeliefState = old_state
        _actual_old_conf: float = old_confidence

        # Persist the state change.
        # write_transaction() uses BEGIN IMMEDIATE to acquire the write lock before the
        # in-tx SELECT, preventing WAL deferred-read-to-write upgrade failures
        # (SQLITE_BUSY_SNAPSHOT) under concurrent writers.
        # NOTE: write_transaction() inside a nested deferred transaction falls back to
        # a SAVEPOINT, which does not acquire an early write lock. There are no current
        # call paths that nest _revise_one() inside an outer deferred transaction.
        now = _now()
        with self.db.write_transaction():
            _intx_row = self.db.execute(
                "SELECT belief_state, confidence FROM assertion WHERE id=?",
                (assertion_id,),
            ).fetchone()
            if _intx_row is None:
                return None, False
            _intx_old_state = BeliefState(_intx_row["belief_state"])
            _intx_old_conf = float(_intx_row["confidence"])
            _actual_old_state = _intx_old_state
            _actual_old_conf = _intx_old_conf

            # OCC (Optimistic Concurrency Control): if the row changed since our
            # pre-tx snapshot, a concurrent writer committed between L302 and here.
            # Abort rather than overwrite a newer committed state with a stale
            # BFS-computed target. Return occ_aborted=True so the BFS caller
            # re-enqueues dependents and does not silently prune the subtree.
            if (_intx_old_state != old_state
                    or abs(_intx_old_conf - old_confidence) >= 0.001):
                return None, True  # Conflict; caller must still enqueue dependents

            # Write immutable field-diff rows before mutating (SO-2, Q4 HIGH).
            # Use in-tx values for both diff detection and old_value_json.
            _rev_rows: list[tuple[str, str, str]] = []
            if new_state != _intx_old_state:
                _rev_rows.append((
                    "belief_state",
                    _json_mod.dumps(_intx_old_state.value),
                    _json_mod.dumps(new_state.value),
                ))
            if abs(new_confidence - _intx_old_conf) >= 0.001:
                _rev_rows.append((
                    "confidence",
                    _json_mod.dumps(_intx_old_conf),
                    _json_mod.dumps(new_confidence),
                ))

            if not _rev_rows:
                # In-tx state already matches the BFS target; nothing to write.
                # Commits an empty transaction (harmless) and returns None.
                return None, False

            self.assertion_store.write_revision_rows(
                assertion_id, _rev_rows, _id(),
                cause.value, "system", run_id, note,
            )
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
                    _intx_old_state.value, new_state.value,
                    _intx_old_conf, new_confidence,
                    note, now,
                ),
            )

        # Post-write oscillation detection: check if this assertion has entered a
        # cyclic belief state pattern (A→B→A) indicating contradictory/unstable evidence.
        # Read-only check; runs outside the write transaction to avoid lock contention.
        _result = RevisionResult(
            assertion_id=assertion_id,
            old_belief_state=_actual_old_state,
            new_belief_state=new_state,
            old_confidence=_actual_old_conf,
            new_confidence=new_confidence,
            cause=cause,
        )
        try:
            if self.assertion_store.detect_oscillation(assertion_id):
                _log.warning(
                    "BeliefRevisionEngine: assertion %s shows oscillating belief state "
                    "(A→B→A cycle detected in revision history) — contradictory or "
                    "unstable evidence network. Affected issues may need manual review.",
                    assertion_id,
                )
                if run_id and self._ledger is not None:
                    self._ledger.append_event(
                        run_id=run_id,
                        event_type=LedgerEventType.SYSTEM_WARNING,
                        summary=(
                            f"Assertion {assertion_id[:8]}… shows oscillating belief state "
                            "(A→B→A cycle in revision history). Evidence may be contradictory "
                            "or temporal/supersession semantics may be missing."
                        ),
                    )
        except Exception as exc:
            _log.debug("Oscillation check failed for %s: %s", assertion_id, exc)
        return _result, False

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

        # _actual_old_* will be overridden with committed in-tx values so that
        # RevisionResult matches what was recorded in the audit trail.
        _actual_old_state: BeliefState = BeliefState(record.belief_state)
        _actual_old_conf: float = record.confidence
        now = _now()

        # write_transaction() uses BEGIN IMMEDIATE to acquire the write lock before the
        # in-tx SELECT, preventing WAL deferred-read-to-write upgrade failures.
        with self.db.write_transaction():
            _fs_intx_row = self.db.execute(
                "SELECT belief_state, confidence FROM assertion WHERE id=?",
                (assertion_id,),
            ).fetchone()
            if _fs_intx_row is None:
                raise ValueError(f"Assertion {assertion_id} disappeared before write")
            _fs_intx_old_state = BeliefState(_fs_intx_row["belief_state"])
            _fs_intx_old_conf = float(_fs_intx_row["confidence"])
            _actual_old_state = _fs_intx_old_state
            _actual_old_conf = _fs_intx_old_conf

            # Write immutable field-diff rows before mutating (SO-2, Q4 HIGH).
            # actor_kind="user" for direct force_state corrections.
            _fs_rev_rows: list[tuple[str, str, str]] = []
            if new_state != _fs_intx_old_state:
                _fs_rev_rows.append((
                    "belief_state",
                    _json_mod.dumps(_fs_intx_old_state.value),
                    _json_mod.dumps(new_state.value),
                ))
            if abs(new_confidence - _fs_intx_old_conf) >= 0.001:
                _fs_rev_rows.append((
                    "confidence",
                    _json_mod.dumps(_fs_intx_old_conf),
                    _json_mod.dumps(new_confidence),
                ))
            if not _fs_rev_rows and cause == RevisionCause.USER_CORRECTION:
                # User explicitly chose this state even though it matches the current DB value.
                # Write a no-op revision row (old_value == new_value) so the supersession
                # recovery path can detect user intent and preserve it when a superseding
                # link later becomes inert. Without this, only the prior BFS-written row
                # would be found, and its actor_kind='system' would permit unwanted recovery.
                _fs_rev_rows.append((
                    "belief_state",
                    _json_mod.dumps(new_state.value),
                    _json_mod.dumps(new_state.value),
                ))
            if _fs_rev_rows:
                self.assertion_store.write_revision_rows(
                    assertion_id, _fs_rev_rows, _id(),
                    cause.value, "user", run_id, note,
                )
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
                    _fs_intx_old_state.value, new_state.value,
                    _fs_intx_old_conf, new_confidence,
                    note, now,
                ),
            )

        result = RevisionResult(
            assertion_id=assertion_id,
            old_belief_state=_actual_old_state,
            new_belief_state=new_state,
            old_confidence=_actual_old_conf,
            new_confidence=new_confidence,
            cause=cause,
        )

        # Propagate to dependents; capture truncation so callers can surface it (SO-2).
        dependents = self.assertion_store.get_dependents(assertion_id)
        if dependents:
            downstream, truncated = self._apply_with_truncation(dependents, cause, run_id, note)
            result.propagated_to = [r.assertion_id for r in downstream]
            result.propagation_truncated = truncated

        return result
